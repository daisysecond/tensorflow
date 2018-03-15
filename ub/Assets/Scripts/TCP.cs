using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Threading;
using System.Net;
using System.Net.Sockets;
using System.IO;
using System;

public class Message {
	public String m;
	public Socket sock;

	public bool IsAction() {
		return m.StartsWith ("a");
	}
	public bool IsReset() {
		return m.StartsWith ("r");
	}
	public bool IsBerry() {
		return m.StartsWith ("b");
	}
	public void Reply(byte[] response) {
		if (response.Length > 0) {
			sock.Send (response);
		}
		sock.Close ();
	}
}

public class TCPQueue {
	private ConcurrentQueue<Message> queue = new ConcurrentQueue<Message>();
	private Thread thread;
	private bool running = false;

	public void OnEnable() {
		running = true;
		thread = new Thread(new ThreadStart(Run));
		thread.Start();
	}
		
	void OnDisable() {
		running = false;
		thread.Join (500);
		if (thread.IsAlive) {
			Debug.LogWarning ("TCP thread did not gracefully exit");
		}
	}

	private void Run() {
		string berryPort = Environment.GetEnvironmentVariable ("BERRY_PORT");
		int port = 8001;
		if (berryPort != null) {
			port = int.Parse (berryPort);
		}
		TcpListener listener = new TcpListener(IPAddress.Parse("127.0.0.1"), port);
		listener.Start ();
		Debug.Log("Listening on localhost:" + port.ToString());

		// Loop
		while (running) {
			if (!listener.Pending()) {
				Thread.Sleep(100);
				continue;
			}
			Socket sock = listener.AcceptSocket();
			sock.Blocking = false;

			byte[] msg = readMessage (sock);
			if (msg != null) {
				String txt = System.Text.Encoding.UTF8.GetString (msg);
				Message m = new Message ();
				m.sock = sock;
				m.m = txt;
				queue.Enqueue (m);
			}
		}

		listener.Stop ();
	}

	private byte[] readMessage(Socket sock) {
		byte[] buf = new byte[10];
		int read = 0;
		bool open = true;
		while (running && open && read < 10) {
			if (!sock.Poll(100, SelectMode.SelectRead)) {
				continue;
			}
			int r = sock.Receive (buf);
			if (r == 0) {
				open = false;
			}
			read += r;
		}
		if (read == 10) {
			return buf;
		}
		return null;
	}

	public int Count {
		get {
			return queue.Count;
		}
	}

	public Message Dequeue() {
		return queue.Dequeue ();
	}
}

public class TCP : MonoBehaviour {
	public Camera mainCamera;

	public Transform bicep;
	public Transform forearm;
	public Transform shoulder;
	public Transform hand;

	private HingeJoint shoulderJoint;
	public float shoulderSpeed = 50;

	private HingeJoint bicepJoint;
	private float bicepSpeed = 50;

	private HingeJoint forearmJoint;
	private float forearmSpeed = 50;

	static TCPQueue queue;
	static Message activeMessage;
	float pauseTime = 0;

	void Start() {
		if (queue == null) {
			queue = new TCPQueue ();
			queue.OnEnable ();
		}

		// If a message is active then it should be a reset message which caused the level to reset. In this case we have
		// now re-created the level so we can reply.
		if (activeMessage != null) {
			activeMessage.Reply (new byte[0]);
			activeMessage = null;
		}

		shoulderJoint = shoulder.GetComponent<HingeJoint> ();
		bicepJoint = bicep.GetComponent<HingeJoint> ();
		forearmJoint = forearm.GetComponent<HingeJoint> ();
	}

	void Update() {
		// Only deque if there is no active message being processed.
		while (activeMessage == null && queue.Count > 0) {
			activeMessage = queue.Dequeue();
			if (activeMessage.IsBerry ()) {
				int bc = int.Parse (activeMessage.m.Substring (1));
				for (int i = 0; i < bc; i += 1) {
					float x = UnityEngine.Random.Range (-2f, 2f);
					float z = 1; //Random.Range (-3f, 3f)
					berries.AddBerry (x, z);
				}
				pauseTime = Time.time + 3f; // 3 secs to settle but run it in 1 sec
				Time.timeScale = 3.0f;
			}
			if (activeMessage.IsReset ()) {
				// This call causes the scene to reset. A new TCP class is created which will respond to the message in
				// the Start() function.
				UnityEngine.SceneManagement.SceneManager.LoadScene (UnityEngine.SceneManagement.SceneManager.GetActiveScene ().buildIndex);
				Time.timeScale = 0;
			}
			if (activeMessage.IsAction()) {
				String msg = activeMessage.m;
				int a = int.Parse (msg.Substring (1, 1)) - 1;
				int b = int.Parse (msg.Substring (2, 1)) - 1;
				int c = int.Parse (msg.Substring (3, 1)) - 1;
				advance (a, b, c);
			}
		}

		// The timer for the active message has expired. 
		if (activeMessage != null && Time.time > pauseTime) {
			Time.timeScale = 0;

			if (activeMessage.IsAction ()) {
				const int FLOAT_LEN = 12;
				float[] f = new float[FLOAT_LEN];
				Vector3 vp = mainCamera.WorldToViewportPoint (hand.position);

				f [0] = hand.position.x;
				f [1] = hand.position.y;
				f [2] = hand.position.z;
				f [3] = vp.x;
				f [4] = vp.y;
				f [5] = vp.z;

				Transform t = berries.ClosestBerry ();
				if (t != null) {
					vp = mainCamera.WorldToViewportPoint (t.position);

					f [6] = t.position.x;
					f [7] = t.position.y;
					f [8] = t.position.z;
					f [9] = vp.x;
					f [10] = vp.y;
					f [11] = vp.z;
				}

				var b = new byte[f.Length * 4 + 4];
				b [0] = (byte)'h';
				b [1] = (byte)'i';
				b [2] = (byte)berries.CollectedCount ();
				Buffer.BlockCopy (f, 0, b, 4, f.Length * 4);

				activeMessage.Reply (b);
			} else {
				activeMessage.Reply (new byte[0]);
			}
			activeMessage = null;
		}
	}

	void FixedUpdate() {
		JointMotor m = shoulderJoint.motor;
		if (shoulderMotor > 0) {
			m.targetVelocity = shoulderSpeed;
		} else if (shoulderMotor < 0) {
			m.targetVelocity = -shoulderSpeed;
		} else {
			m.targetVelocity = 0f;
		}
		shoulderJoint.motor = m;

		m = bicepJoint.motor;
		if (bicepMotor > 0) {
			m.targetVelocity = -bicepSpeed;
		} else if (bicepMotor < 0) {
			m.targetVelocity = bicepSpeed;
		} else {
			m.targetVelocity = 0f;
		}
		bicepJoint.motor = m;

		m = forearmJoint.motor;
		if (forearmMotor > 0) {
			m.targetVelocity = -forearmSpeed;
		} else if (forearmMotor < 0) {
			m.targetVelocity = forearmSpeed;
		} else {
			m.targetVelocity = 0f;
		}
		forearmJoint.motor = m;
	}

	public BerriesController berries;

	float bicepMotor = 0;
	float forearmMotor = 0;
	float shoulderMotor = 0;

	void advance(int a, int b, int c) {
		shoulderMotor = a;
		bicepMotor = b;
		forearmMotor = c;

		pauseTime = Time.time + 0.02f;
		Time.timeScale = 2;
	}
}


public class ConcurrentQueue<T> : ICollection, IEnumerable<T>
{
	private readonly Queue<T> _queue;

	public ConcurrentQueue()
	{
		_queue = new Queue<T>();
	}

	public IEnumerator<T> GetEnumerator()
	{
		lock (SyncRoot)
		{
			foreach (var item in _queue)
			{
				yield return item;
			}
		}
	}

	IEnumerator IEnumerable.GetEnumerator()
	{
		return GetEnumerator();
	}

	public void CopyTo(Array array, int index)
	{
		lock (SyncRoot)
		{
			((ICollection)_queue).CopyTo(array, index);
		}
	}

	public int Count
	{
		get
		{ 
			// Assumed to be atomic, so locking is unnecessary
			return _queue.Count;
		}
	}

	public object SyncRoot
	{
		get { return ((ICollection)_queue).SyncRoot; }
	}

	public bool IsSynchronized
	{
		get { return true; }
	}

	public void Enqueue(T item)
	{
		lock (SyncRoot)
		{
			_queue.Enqueue(item);
		}
	}

	public T Dequeue()
	{
		lock(SyncRoot)
		{
			return _queue.Dequeue();
		}
	}

	public T Peek()
	{
		lock (SyncRoot)
		{
			return _queue.Peek();
		}
	}

	public void Clear()
	{
		lock (SyncRoot)
		{
			_queue.Clear();
		}
	}
}
