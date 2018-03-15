using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Text;
using System.IO;


public class VisionLoop : MonoBehaviour {
	static private int captureCount = 0;

	public Camera rgbCamera;
	public Camera depthCamera;

	public bool runLoop = false;

	private bool isDone = false;
	private float loopTime;

	const int maxBerries = 4;
	public BerriesController berries;

	void Start () {
		if (!runLoop) {
			return;
		}

		// Create count berries, randomizing position and rotation.
		int count = Random.Range(0, maxBerries);
		for (int i = 0; i < count; i++) {
			float x = Random.Range (-2f, 2f);
			float z = Random.Range (-1f, 3f);
			berries.AddBerry (x, z);
		}

		// Purturbate:
		//    Leafs
		//    Camera
		//    Arm
		//    Lighting

		// Settle for a random time
		loopTime = 3f; //Random.Range (2f, 5f);
	}

	static Color c = new Color(0f, 0f, 1f, 0.5f);

	void OnGUI() {
		foreach (Rect rect in berries.GetBerryRects(rgbCamera)) {
			Texture2D texture = new Texture2D(1, 1);
			texture.SetPixel(0,0, c);
			texture.Apply();
			GUI.skin.box.normal.background = texture;
			GUI.Box(rect, GUIContent.none);
		}
	}

	void Update () {
		float t = Time.timeSinceLevelLoad;
		if (!isDone && runLoop && t > loopTime) {
			capture ();
			// Reset the scene.
			UnityEngine.SceneManagement.SceneManager.LoadScene(UnityEngine.SceneManagement.SceneManager.GetActiveScene().buildIndex); 
			isDone = true;
		}
	
	}


	const string dir = "/home/adam/Desktop/b/";

	int imageWidth = 299;
	int imageHeight = 299;

	void captureCamera(Camera c, RenderTexture rt, Texture2D screenShot) {
		c.targetTexture = rt;
		c.Render();
		RenderTexture.active = rt;
		screenShot.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
		c.targetTexture = null;
		RenderTexture.active = null;
	}

	const int FLOAT_PER_CELL = 5;

	void setf(float[] data, int x, int y, int index, float v) {
		int i = y * HORIZONTAL_CELLS * FLOAT_PER_CELL +
		        x * FLOAT_PER_CELL +
		        index;
		data [i] = v;
	}

	void capture() {
		print ("Capture " + captureCount);

		RenderTexture rt = new RenderTexture(imageWidth, imageHeight, 24);
		Texture2D screenShot = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
		captureCamera (rgbCamera, rt, screenShot);
		byte[] bytes = screenShot.EncodeToPNG();
		System.IO.File.WriteAllBytes(dir + captureCount + ".png", bytes);

		captureCamera (depthCamera, rt, screenShot);
		bytes = screenShot.EncodeToPNG();
		System.IO.File.WriteAllBytes(dir + captureCount + "d.png", bytes);

		Destroy(rt);

		Rect[] rects = berries.GetBerryRects (rgbCamera);

		float cellWidth = rgbCamera.pixelWidth / (float) HORIZONTAL_CELLS;
		float cellHeight = rgbCamera.pixelHeight / (float) VERTICAL_CELLS;

		float[] data = new float[HORIZONTAL_CELLS * VERTICAL_CELLS * FLOAT_PER_CELL];

		for (int y = 0; y < VERTICAL_CELLS; y += 1) {
			for (int x = 0; x < HORIZONTAL_CELLS; x += 1) {
				// For each cell in the grid test if it overlaps any berries.
				// If it does then determine the best berry by picking the closest one.

				Rect cell = new Rect (x * cellWidth, y * cellHeight, cellWidth, cellHeight);

				Rect bestBerry = new Rect();
				bool found = false;
				float bestDistance = float.MaxValue;
				foreach (Rect berryRect in rects) {
					if (!berryRect.Overlaps (cell)) {
						continue;
					}
					float d = Vector2.Distance (berryRect.center, cell.center);
					if (d < bestDistance) {
						found = true;
						bestDistance = d;
						bestBerry = berryRect;
					}
				}
				setf (data, x, y, 0, found ? 1f : 0f);

				if (found) {
					setf (data, x, y, 1, cell.x - bestBerry.x); // left
					setf (data, x, y, 2, cell.xMax - bestBerry.xMax); // right
					setf (data, x, y, 3, cell.y - bestBerry.y); // bottom
					setf (data, x, y, 4, cell.yMax - bestBerry.yMax); // top
				}
			}
		}
		//byte[] data = new byte[4];

		// has_berry
		//Transform closestBerry = berries.ClosestBerry();
		//data [0] = closestBerry == null ? (byte) 0 : (byte) 1;
		//if (closestBerry != null) {
		//	data [1] = bytePosition(closestBerry.position.x);
		//	data [2] = bytePosition(closestBerry.position.y);
		//	data [3] = bytePosition(closestBerry.position.z);
		//}
		
		// Save arm position
		StringBuilder csv = new StringBuilder();
		int count = 0;
		foreach (float f in data) {
			csv.Append (f.ToString());
			csv.Append (",");
			if (count > 0 && ((count + 1) % (HORIZONTAL_CELLS * FLOAT_PER_CELL)) == 0) {
				csv.Append ("\n");	
			}
			count += 1;
		}
		File.WriteAllText(dir + captureCount + ".csv", csv.ToString());

		captureCount += 1;
	}

	// Assumes table -5 to 5
	byte bytePosition(float p) {
		// Convert to [0, 1]
		float norm = (p / 10f) + 0.5f;
		byte floor = (byte) Mathf.Floor (norm * 256);
		return floor;
	}

	const int HORIZONTAL_CELLS = 8;
	const int VERTICAL_CELLS = 8;

}
