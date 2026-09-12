#!/usr/bin/env python3
import sys
import argparse
import copy
import cv2
import rosbag
import rospy
import genpy
import av
from cv_bridge import CvBridge
import data_msgs.msg as msg_type
from perception_msgs.msg import KinectImages
from data_pipeline.tools.robot_message_conversion import (ObservationConversion, ActionConversion,
                                                          ConfigConversion, TactileConversion,
                                                          EventConversion)
from mcap.reader import make_reader

decoder_factories = []
from mcap_ros1.decoder import DecoderFactory as ROS1DecoderFactory
decoder_factories.append(ROS1DecoderFactory())


def parse(format: str):
    codec, is_keyframe, pts, dts, ts = format.split(';')
    pts = int(pts)
    dts = int(dts)
    ts = int(ts)
    is_keyframe = bool(int(is_keyframe))
    return codec, is_keyframe, pts, dts, ts


def flush_decode(codec):
    return decode(codec, None, 0, 0, False)


def decode(codec, data, pts, dts, is_key_frame):
    try:
        if data is None:
            frames = codec.decode(None)
        else:
            packet = av.Packet(data)
            packet.pts = pts
            packet.dts = dts
            frames = codec.decode(packet)
        decoded_frames = []
        for frame in frames:
            img = frame.to_ndarray(format='bgr24')
            decoded_frames.append(img)
        return decoded_frames
    except av.error.EOFError:
        print("Decode finished: ", data is None)
    except av.error.InvalidDataError as e:
        print(f"Decode error: {e}")
    return []


def create_decoder(is_h264):
    if is_h264:
        decoder = av.CodecContext.create('h264', 'r')
    else:
        decoder = av.CodecContext.create('hevc', 'r')
    return decoder


def decode_all_frames(frame_id: str, msgs: list):
    decoded_frames = []
    bridge = CvBridge()
    sorted_msgs = {}
    pts_dict = {}
    for msg in msgs:
        codec, is_keyframe, pts, dts, ts = parse(msg.format)
        sorted_msgs[dts] = (codec, is_keyframe, pts, dts, ts, msg.data)
        pts_dict[pts] = ts

    codec = [elem[0] for elem in sorted_msgs.values()][0]
    decoder = create_decoder(codec == "h264")
    sorted_msgs = dict(sorted(sorted_msgs.items()))
    decoded_pts = 0
    for key in sorted_msgs:
        (codec, is_keyframe, pts, dts, ts, data) = sorted_msgs[key]
        if len(data) == 0:
            continue
        images = decode(decoder, data, pts, dts, is_keyframe)
        for image in images:
            msg = bridge.cv2_to_compressed_imgmsg(image, "jpg")
            ts = pts_dict[decoded_pts]
            msg.header.stamp = rospy.Time.from_sec(ts / 1000000000.0)
            msg.header.frame_id = frame_id
            decoded_pts += 1
            decoded_frames.append(msg)
    remaining_frames = flush_decode(decoder)
    for image in remaining_frames:
        msg = bridge.cv2_to_compressed_imgmsg(image, "jpg")
        ts = pts_dict[decoded_pts]
        msg.header.stamp = rospy.Time.from_sec(ts / 1000000000.0)
        msg.header.frame_id = frame_id
        decoded_pts += 1
        decoded_frames.append(msg)
    return decoded_frames


def align_msgs(target_msg, ref_msg, insert_previous_data):
    target_size = len(target_msg)
    ref_size = len(ref_msg)
    if target_size == 0 or target_size >= ref_size:
        return target_msg
    i = 0
    j = 0
    print(f'Insert {ref_size-target_size} to {target_msg[i].header.frame_id}')
    added_msg = type(target_msg[i])()
    if insert_previous_data:
        added_msg = target_msg[i]
    else:
        added_msg.header = target_msg[i].header
    target_stamp = target_msg[i].header.stamp
    ret_msg = []
    while j < ref_size :
        ref_stamp = ref_msg[j].header.stamp
        if target_stamp <= ref_stamp and i < target_size:
            ret_msg.append(target_msg[i])
            if insert_previous_data:
                added_msg = target_msg[i]
            i += 1
            if i < target_size:
                target_stamp = target_msg[i].header.stamp
        else:
            added_msg.header.stamp = ref_stamp
            ret_msg.append(copy.deepcopy(added_msg))
        j += 1
    if i < target_size:
        ret_msg[(i - target_size):] = target_msg[(i - target_size):]

    return ret_msg


class RobotRosbagDecoder():
    def __init__(self):
        self.observations = []
        self.actions = []
        self.config = None
        self.frequency = None
        self.duration = None
        self.size = sys.maxsize

    def decode(self, bag_path, start_sec=0.0, end_sec=0.0, insert_previous_data=False):
        if bag_path.endswith('.mcap'):
            self.decode_mcap(bag_path, start_sec, end_sec, insert_previous_data)
        else:
            self.decode_rosbag(bag_path, start_sec, end_sec, insert_previous_data)

    def decode_mcap(self, bag_path, start_sec=0.0, end_sec=0.0, insert_previous_data=False):
        with open(bag_path, "rb") as f:
            reader = make_reader(f, decoder_factories=decoder_factories)

            summary = reader.get_summary()
            print(summary)

            self.observations = []
            self.actions = []
            print(f'Start decoding {bag_path}...')
            bag_start_t = summary.statistics.message_start_time / 1e9
            end_t = None
            start_t = None
            cam_dict = dict()
            color_dict = dict()
            depth_dict = dict()
            ir_dict = dict()
            obs_dict = dict()
            act_dict = dict()
            tactile_dict = dict()
            event_dict = dict()
            config_msg = None
            if start_sec > 0.0:
                start_t = genpy.Time.from_sec(bag_start_t + start_sec)
            if end_sec > start_sec:
                end_t = genpy.Time.from_sec(bag_start_t + end_sec)
            use_ffmpeg_camera = False
            use_ffmpeg_chain_color = False
            for _, channel, _, decoded_msg in reader.iter_decoded_messages(start_time=start_t, end_time=end_t):
                topic = channel.topic
                msg = decoded_msg
                if topic.split('/')[-1] == 'config':
                    config_msg = msg
                    continue
                if '/camera_image' in topic:
                    if (not use_ffmpeg_camera) and msg.format != 'jpg' and msg.format != 'png':
                        use_ffmpeg_camera = True
                    if msg.header.frame_id not in cam_dict.keys():
                        cam_dict[msg.header.frame_id] = [msg]
                    else:
                        cam_dict[msg.header.frame_id].append(msg)
                    continue
                if '/color_image' in topic:
                    if (not use_ffmpeg_chain_color) and msg.format != 'jpg' and msg.format != 'png':
                        use_ffmpeg_chain_color = True
                    if msg.header.frame_id not in color_dict.keys():
                        color_dict[msg.header.frame_id] = [msg]
                    else:
                        color_dict[msg.header.frame_id].append(msg)
                    continue
                if '/depth_image' in topic:
                    if msg.header.frame_id not in depth_dict.keys():
                        depth_dict[msg.header.frame_id] = [msg]
                    else:
                        depth_dict[msg.header.frame_id].append(msg)
                    continue
                if '/ir_image' in topic:
                    if msg.header.frame_id not in ir_dict.keys():
                        ir_dict[msg.header.frame_id] = [msg]
                    else:
                        ir_dict[msg.header.frame_id].append(msg)
                    continue
                if '/observation' in topic:
                    if msg.header.frame_id not in obs_dict.keys():
                        obs_dict[msg.header.frame_id] = [msg]
                    else:
                        obs_dict[msg.header.frame_id].append(msg)
                    continue
                if '/action' in topic:
                    if msg.header.frame_id not in act_dict.keys():
                        act_dict[msg.header.frame_id] = [msg]
                    else:
                        act_dict[msg.header.frame_id].append(msg)
                    continue
                if '/tactile' in topic:
                    if hasattr(msg, 'layout'):
                        if len(msg.layout.dim) == 0:
                            print(f'Tactile message from {topic} can not be parsed')
                            continue
                        name = msg.layout.dim[0].label
                        if name not in tactile_dict.keys():
                            tactile_dict[name] = [msg]
                        else:
                            tactile_dict[name].append(msg)
                    else:
                        print(f'Unsupported tactile message type {type(msg)}')
                        continue
                if '/event' in topic:
                    if msg.header.frame_id not in event_dict.keys():
                        event_dict[msg.header.frame_id] = [msg]
                    else:
                        event_dict[msg.header.frame_id].append(msg)
                    continue

            if use_ffmpeg_camera:
                for key in list(cam_dict.keys()):
                    cam_dict[key] = decode_all_frames(key, cam_dict[key])
            if use_ffmpeg_chain_color:
                for key in list(color_dict.keys()):
                    color_dict[key] = decode_all_frames(key, color_dict[key])
            for val in obs_dict.values():
                self.size = min(self.size, len(val))
            if self.size == sys.maxsize:
                self.size = 0
            ref_msg = list(obs_dict.values())[0]
            for key in act_dict.keys():
                act_dict[key] = align_msgs(act_dict[key], ref_msg, insert_previous_data)
            for key in cam_dict.keys():
                cam_dict[key] = align_msgs(cam_dict[key], ref_msg, insert_previous_data)
            for key in color_dict.keys():
                color_dict[key] = align_msgs(color_dict[key], ref_msg, insert_previous_data)
            for key in depth_dict.keys():
                depth_dict[key] = align_msgs(depth_dict[key], ref_msg, insert_previous_data)
            last_event_index = None
            for key in event_dict.keys():
                event_dict[key] = align_msgs(event_dict[key], ref_msg, False)
                if insert_previous_data:
                    cur_event = msg_type.Event()
                    for i in range(-1, -len(event_dict[key]) - 1, -1):
                        if event_dict[key][i].event_type == '' and event_dict[key][i].event_detail == '':
                            event_dict[key][i].event_type = cur_event.event_type
                            event_dict[key][i].event_detail = cur_event.event_detail
                        else:
                            cur_event = event_dict[key][i]
                            if last_event_index is None:
                                last_event_index = i + len(event_dict[key])
            if last_event_index is not None:
                print(f'Crop size from {self.size} to last event index {last_event_index}')
                self.size = last_event_index
            for i in range(self.size):
                obs_msg = msg_type.RobotObservation()
                for key in cam_dict.keys():
                    obs_msg.camera_images.images.append(cam_dict[key][i])

                for key in color_dict.keys():
                    img = KinectImages()
                    img.header = color_dict[key][i].header
                    img.color_image = color_dict[key][i]
                    if key in depth_dict.keys():
                        img.depth_image = depth_dict[key][i]
                    if key in ir_dict.keys():
                        img.ir_image = ir_dict[key][i]
                    obs_msg.chain_images.chain_images.append(img)

                for key in obs_dict.keys():
                    obs_msg.header.stamp = obs_dict[key][i].header.stamp
                    obs_msg.component_observations.append(obs_dict[key][i])
                cur_obs = ObservationConversion.from_robot_message(obs_msg)

                for key in tactile_dict.keys():
                    if i > len(tactile_dict[key]):
                        print(f'Tactile {key} has no enough data {len(tactile_dict[key])}')
                        continue
                    cur_obs.tactiles.append(TactileConversion.from_robot_message(tactile_dict[key][i]))

                for key in event_dict.keys():
                    cur_obs.event = EventConversion.from_robot_message(event_dict[key][i])

                self.observations.append(cur_obs)

                act_msg = msg_type.RobotAction()
                act_msg.header = obs_msg.header
                for key in act_dict.keys():
                    act_msg.component_actions.append(act_dict[key][i])
                self.actions.append(ActionConversion.from_robot_message(act_msg))
            if config_msg is not None:
                self.config = ConfigConversion.from_robot_message(config_msg)

            if self.size > 0:
                if end_sec > start_sec:
                    self.duration = end_sec - start_sec
                else:
                    self.duration = self.observations[-1].timestamp - self.observations[0].timestamp
                self.frequency = self.size / self.duration
            print(
                (f'Finish decoding. Frequency:{self.frequency}; Duration:{self.duration};'
                f' Observation size:{len(self.observations)}; Action size:{len(self.actions)}'))

    def decode_rosbag(self, bag_path, start_sec=0.0, end_sec=0.0, insert_previous_data=False):
        bag = rosbag.Bag(bag_path)
        print(bag.get_type_and_topic_info())
        self.observations = []
        self.actions = []
        print(f'Start decoding {bag_path}...')
        bag_start_t = bag.get_start_time()
        end_t = None
        start_t = None
        cam_dict = dict()
        color_dict = dict()
        depth_dict = dict()
        ir_dict = dict()
        obs_dict = dict()
        act_dict = dict()
        tactile_dict = dict()
        event_dict = dict()
        config_msg = None
        if start_sec > 0.0:
            start_t = genpy.Time.from_sec(bag_start_t + start_sec)
        if end_sec > start_sec:
            end_t = genpy.Time.from_sec(bag_start_t + end_sec)
        use_ffmpeg_camera = False
        use_ffmpeg_chain_color = False
        for topic, msg, _ in bag.read_messages(start_time=start_t,
                                               end_time=end_t):
            if topic.split('/')[-1] == 'config':
                config_msg = msg
                continue
            if '/camera_image' in topic:
                if (not use_ffmpeg_camera) and msg.format != 'jpg' and msg.format != 'png':
                    use_ffmpeg_camera = True
                if msg.header.frame_id not in cam_dict.keys():
                    cam_dict[msg.header.frame_id] = [msg]
                else:
                    cam_dict[msg.header.frame_id].append(msg)
                continue
            if '/color_image' in topic:
                if (not use_ffmpeg_chain_color) and msg.format != 'jpg' and msg.format != 'png':
                    use_ffmpeg_chain_color = True
                if msg.header.frame_id not in color_dict.keys():
                    color_dict[msg.header.frame_id] = [msg]
                else:
                    color_dict[msg.header.frame_id].append(msg)
                continue
            if '/depth_image' in topic:
                if msg.header.frame_id not in depth_dict.keys():
                    depth_dict[msg.header.frame_id] = [msg]
                else:
                    depth_dict[msg.header.frame_id].append(msg)
                continue
            if '/ir_image' in topic:
                if msg.header.frame_id not in ir_dict.keys():
                    ir_dict[msg.header.frame_id] = [msg]
                else:
                    ir_dict[msg.header.frame_id].append(msg)
                continue
            if '/observation' in topic:
                if msg.header.frame_id not in obs_dict.keys():
                    obs_dict[msg.header.frame_id] = [msg]
                else:
                    obs_dict[msg.header.frame_id].append(msg)
                continue
            if '/action' in topic:
                if msg.header.frame_id not in act_dict.keys():
                    act_dict[msg.header.frame_id] = [msg]
                else:
                    act_dict[msg.header.frame_id].append(msg)
                continue
            if '/tactile' in topic:
                if hasattr(msg, 'layout'):
                    if len(msg.layout.dim) == 0:
                        print(f'Tactile message from {topic} can not be parsed')
                        continue
                    name = msg.layout.dim[0].label
                    if name not in tactile_dict.keys():
                        tactile_dict[name] = [msg]
                    else:
                        tactile_dict[name].append(msg)
                else:
                    print(f'Unsupported tactile message type {type(msg)}')
                    continue
            if '/event' in topic:
                if msg.header.frame_id not in event_dict.keys():
                    event_dict[msg.header.frame_id] = [msg]
                else:
                    event_dict[msg.header.frame_id].append(msg)
                continue

        if use_ffmpeg_camera:
            for key in list(cam_dict.keys()):
                cam_dict[key] = decode_all_frames(key, cam_dict[key])
        if use_ffmpeg_chain_color:
            for key in list(color_dict.keys()):
                color_dict[key] = decode_all_frames(key, color_dict[key])
        for val in obs_dict.values():
            self.size = min(self.size, len(val))
        if self.size == sys.maxsize:
            self.size = 0
        ref_msg = list(obs_dict.values())[0]
        for key in act_dict.keys():
            act_dict[key] = align_msgs(act_dict[key], ref_msg, insert_previous_data)
        for key in cam_dict.keys():
            cam_dict[key] = align_msgs(cam_dict[key], ref_msg, insert_previous_data)
        for key in color_dict.keys():
            color_dict[key] = align_msgs(color_dict[key], ref_msg, insert_previous_data)
        for key in depth_dict.keys():
            depth_dict[key] = align_msgs(depth_dict[key], ref_msg, insert_previous_data)
        last_event_index = None
        for key in event_dict.keys():
            event_dict[key] = align_msgs(event_dict[key], ref_msg, False)
            if insert_previous_data:
                cur_event = msg_type.Event()
                for i in range(-1, -len(event_dict[key]) - 1, -1):
                    if event_dict[key][i].event_type == '' and event_dict[key][i].event_detail == '':
                        event_dict[key][i].event_type = cur_event.event_type
                        event_dict[key][i].event_detail = cur_event.event_detail
                    else:
                        cur_event = event_dict[key][i]
                        if last_event_index is None:
                            last_event_index = i + len(event_dict[key])
        if last_event_index is not None:
            print(f'Crop size from {self.size} to last event index {last_event_index}')
            self.size = last_event_index
        for i in range(self.size):
            obs_msg = msg_type.RobotObservation()
            for key in cam_dict.keys():
                obs_msg.camera_images.images.append(cam_dict[key][i])

            for key in color_dict.keys():
                img = KinectImages()
                img.header = color_dict[key][i].header
                img.color_image = color_dict[key][i]
                if key in depth_dict.keys():
                    img.depth_image = depth_dict[key][i]
                if key in ir_dict.keys():
                    img.ir_image = ir_dict[key][i]
                obs_msg.chain_images.chain_images.append(img)

            for key in obs_dict.keys():
                obs_msg.header.stamp = obs_dict[key][i].header.stamp
                obs_msg.component_observations.append(obs_dict[key][i])
            cur_obs = ObservationConversion.from_robot_message(obs_msg)

            for key in tactile_dict.keys():
                if i > len(tactile_dict[key]):
                    print(f'Tactile {key} has no enough data {len(tactile_dict[key])}')
                    continue
                cur_obs.tactiles.append(TactileConversion.from_robot_message(tactile_dict[key][i]))

            for key in event_dict.keys():
                cur_obs.event = EventConversion.from_robot_message(event_dict[key][i])

            self.observations.append(cur_obs)

            act_msg = msg_type.RobotAction()
            act_msg.header = obs_msg.header
            for key in act_dict.keys():
                act_msg.component_actions.append(act_dict[key][i])
            self.actions.append(ActionConversion.from_robot_message(act_msg))
        if config_msg is not None:
            self.config = ConfigConversion.from_robot_message(config_msg)

        if self.size > 0:
            if end_sec > start_sec:
                self.duration = end_sec - start_sec
            else:
                self.duration = self.observations[-1].timestamp - self.observations[0].timestamp
            self.frequency = self.size / self.duration
        print(
            (f'Finish decoding. Frequency:{self.frequency}; Duration:{self.duration};'
             f' Observation size:{len(self.observations)}; Action size:{len(self.actions)}'))

    def find_indices_of_event(self):
        indices = []
        for i in range(len(self.observations)):
            if self.observations[i].event is None:
                continue
            indices.append(i)
        return indices


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters from command line')
    parser.add_argument('--bag_path', type=str, help='rosbag的绝对路径')
    parser.add_argument('--start_sec', type=float, help='开始读取的时间', default=0.0)
    parser.add_argument('--end_sec', type=float, help='结束读取的时间', default=0.0)
    parser.add_argument('--insert_previous_data', action='store_true', default=False,
                        help='Insert with previous data if True')
    _ARGS, _ = parser.parse_known_args()
    decoder = RobotRosbagDecoder()
    decoder.decode(_ARGS.bag_path, _ARGS.start_sec, _ARGS.end_sec, _ARGS.insert_previous_data)
    print(f'events indices: {decoder.find_indices_of_event()}')
    for obs in decoder.observations:
        for img in obs.chain_images:
            if img.color_image is None:
                continue
            cv2.imshow(f'chain images {img.frame_id}', img.color_image)
        for img in obs.camera_images:
            if img.image is None:
                continue
            cv2.imshow(f'camera images {img.frame_id}', img.image)
        if cv2.waitKey(5) == ord('q'):
            break
        for tactile in obs.tactiles:
            print(f'{tactile.name}:\n{tactile.data}')
        if obs.event is not None:
            print(obs.event)
