using UnityEngine;
using System.Collections;

using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;

public class UDPReceive : MonoBehaviour
{
	// receiving Thread
	Thread receiveThread;

	// udpclient object
	UdpClient client;

	public float heightOffset;
	public int cnter = 0;
	public string eR = "";
	public float eX;
	public float eY;
	public float eZ;
	public GameObject player;

	// public
	// public string IP = "127.0.0.1"; default local
	public int port;
	// define > init

	// infos
	public string lastReceivedUDPPacket = "";
	public string allReceivedUDPPackets = "";
	// clean up this from time to time!


	// start from shell
	private static void Main()
	{
		UDPReceive receiveObj = new UDPReceive();
		receiveObj.init();

		string text = "";
		do
		{
			text = Console.ReadLine();
		} while (!text.Equals("exit"));
	}
	// start from unity3d
	public void Start()
	{

		init();
	}

	void Update()
	{
		Vector3 roboPos = new Vector3(player.transform.position.x, player.transform.position.y+eY, player.transform.position.z);
		player.transform.SetPositionAndRotation(roboPos, player.transform.rotation);
	}

	//// OnGUI
	//void OnGUI()
	//{
	//    Rect rectObj=new Rect(40,10,200,400);
	//        GUIStyle style = new GUIStyle();
	//            style.alignment = TextAnchor.UpperLeft;
	//    GUI.Box(rectObj,"# UDPReceive\n127.0.0.1 "+port+" #\n"
	//                + "shell> nc -u 127.0.0.1 : "+port+" \n"
	//                + "\nLast Packet: \n"+ lastReceivedUDPPacket
	//                + "\n\nAll Messages: \n"+allReceivedUDPPackets
	//            ,style);
	//}

	// init
	private void init()
	{
		// Endpunkt definieren, von dem die Nachrichten gesendet werden.
		print("UDPSend.init()");
		// ----------------------------
		// Abhören
		// ----------------------------
		// Lokalen Endpunkt definieren (wo Nachrichten empfangen werden).
		// Einen neuen Thread für den Empfang eingehender Nachrichten erstellen.
		receiveThread = new Thread(new ThreadStart(ReceiveData));
		receiveThread.IsBackground = true;
		receiveThread.Start();

	}


	// receive thread
	private void ReceiveData()
	{
		client = new UdpClient(port);
		while (true)
		{

			try
			{
				// Bytes empfangen.
				IPEndPoint anyIPR = new IPEndPoint(IPAddress.Any, 0);
				byte[] dataR = client.Receive(ref anyIPR);
				string textr = Encoding.UTF8.GetString(dataR);
				string[] message = textr.Split(",");
				eY = float.Parse(message[2]); 
				eR = textr;
				print(eR); 
			}
			catch
			{
				//print (err.ToString ());
			}
		}
	}
}