import argparse
import sys
import time
import socket
import numpy as np
import aria.sdk as aria
import cv2

from common import update_iptables

from visualizer import AriaVisualizer, AriaVisualizerStreamingClientObserver, hand_pose

# ------------------------------------------------------------------
def update_iptables():
    import subprocess, shlex
    try:
        cmd = "sudo iptables -A INPUT -p udp --dport 9999 -j ACCEPT"
        subprocess.run(shlex.split(cmd), check=True)
        print("iptables updated to allow streaming data.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to update iptables: {e}")

# -------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aria RGB stream + UDP forwarding with image shrinking"
        )
    parser.add_argument( "--device-ip", help="Aria's IP address (WiFi streaming)"
    )
    parser.add_argument( "--interface", choices=["usb", "wifi"], default="usb",
        help="Streaming Interface (default: usb)."
    )
    parser.add_argument( "--update_iptables", action="store_true", default=False,
        help="Update iptables (Linux only)."
    )
    parser.add_argument( "--send_udp", action="store_true", 
        help="If set, will forward images over UDP."
    )
    parser.add_argument( "--remote_ip", type=str, default="127.0.0.1", #"169.254.140.163",
        help="UDP forwarding IP (default: 127.0.0.1)."
    )
    parser.add_argument( "--remote_port", type=int, default=8899,
        help="UDP forwarding port (default: 8899)."
    )
    parser.add_argument( "--shrink_factor", type=float, default=0.25,
        help="Frame's shrink factor (default: 0.25)."
    )
    return parser.parse_args()


# --------------------------------------------------------------------
def main():
    args = parse_args()
    aria.set_log_level(aria.Level.Info)
        
        
    device_client = aria.DeviceClient()
    client_config = aria.DeviceClientConfig()
    device_client.set_client_config(client_config)

    # 2. connect to device
    device = device_client.connect()

    # 3. retrive streaming manager and streaming client.
    streaming_manager = device.streaming_manager
    streaming_client = streaming_manager.streaming_client

    # 4 set custom config for streaming
    streaming_config = aria.StreamingConfig()


    #  Set interface  (default:Wifi)
    if args.interface == "usb":
        streaming_config.streaming_interface = aria.StreamingInterface.Usb

    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config


    # 5 start streaming
    streaming_manager.start_streaming()
    streaming_state = streaming_manager.streaming_state
    print(f"Streaming state: {streaming_state}")


    # UDP stream of shrunken frames.
    #-------------------------------------
    udp_socket = None
    if args.send_udp:
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"UDP forwarding enabled. Target: {args.remote_ip}:{args.remote_port}")


    # 7. Create the visualizer observer and attach the streaming client
    aria_visualizer = AriaVisualizer()
    aria_visualizer_streaming_client_observer = AriaVisualizerStreamingClientObserver(
        aria_visualizer
    )
    
    
    # 8. set the Observer streaming client to the Aria Visualizer.
    aria_visualizer.set_single_camera_observer( udp_socket=udp_socket, remote_ip=args.remote_ip,
        remote_port=args.remote_port, shrink_factor=args.shrink_factor)
    
    #
    # read hand icons.
    #
    # grab = cv2.imread("Hand_Grab.png") 
    # mid = cv2.imread("Hand_Mid.png") 
    # full = cv2.imread("Hand_Full_Open.png") 
    # cv2.waitKey(0)
    # aria_visualizer.singleCameraObserver.icons[hand_pose.GRABBING] = grab
    # aria_visualizer.singleCameraObserver.icons.icons[hand_pose.MID] = mid
    # aria_visualizer.singleCameraObserver.icons.icons[hand_pose.FULL_OPEN] = full
    
    
    streaming_client.set_streaming_client_observer(
        aria_visualizer_streaming_client_observer
    )
    streaming_client.subscribe()


    # 8. Visualize the streaming data until we close the window
    aria_visualizer.render_loop()

    streaming_client.unsubscribe()
    streaming_manager.stop_streaming()
    device_client.disconnect(device)

if __name__ == "__main__":
    main()
