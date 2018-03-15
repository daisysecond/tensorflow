using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BerriesController : MonoBehaviour {
	public new Camera camera;
	public Transform strawberryPrefab;
	private List<Transform> strawberries = new List<Transform>();

	int collectedCount;

	void Start() {
		collectedCount = 0;
	}

	public void AddBerry (float x, float z) {
		// Randomize the objects rotation
		Quaternion rotation = new Quaternion();
		Vector3 toDirection = new Vector3 (Random.Range (0f, 1f), Random.Range (0f, 0.8f), Random.Range (0f, 1f));
		rotation.SetFromToRotation (Vector3.up, toDirection);
		var b = Instantiate (strawberryPrefab, transform.up, rotation);
		strawberries.Add (b);
		b.transform.parent = gameObject.transform;

		b.transform.localPosition = new Vector3 (x, 2, z);
	}

	public void CollectBerry(Transform t) {
		RemoveAt (strawberries.IndexOf (t), t);
		collectedCount += 1;
	}

	private void RemoveAt(int index, Transform t) {
		strawberries.RemoveAt (index);
		Destroy (t.gameObject);
	}

	public Transform ClosestBerry() {
		Transform closestBerry = null;
		float distance = 0f;
		foreach (Transform sb in strawberries) {
			// Test if berry has fallen off the table
			if (sb.position.y < 0) {
				continue;
			}
			float d = (sb.position - camera.gameObject.transform.position).sqrMagnitude;
			if (distance == 0f || d < distance) {
				distance = d;
				closestBerry = sb;
			}
		}
		return closestBerry;
	}

	public void Update() {
		// Destroy any that fall off the table
		for (int i = strawberries.Count - 1; i >= 0; i -= 1) {
			Transform t = strawberries [i];
			if (t.transform.position.y < 0) {
				RemoveAt (i, t);
			}
		}
	}

	public int CollectedCount() {
		return collectedCount;
	}

	public Rect[] GetBerryRects(Camera camera) {
		Rect[] ret = new Rect[strawberries.Count];
		for (int i = 0; i < strawberries.Count; i += 1) {
			Transform t = strawberries [i];
			ret [i] = GUIRectWithObject (t.gameObject, camera);
		}
		return ret;
	}

	public static Vector2 WorldToGUIPoint(Vector3 world, Camera camera)
	{
		Vector2 screenPoint = camera.WorldToScreenPoint(world);
		screenPoint.y = (float) Screen.height - screenPoint.y;
		return screenPoint;
	}

	static Vector3 size = new Vector3 (0.25f, 0.15f, 0.25f);

	public static Rect GUIRectWithObject(GameObject go, Camera camera)
	{
		Vector3 cen = go.GetComponent<Renderer>().bounds.center;
		Vector3 p1 = cen + size;
		Vector3 p2 = cen - size;

		Vector2[] extentPoints = new Vector2[2]
		{
			WorldToGUIPoint(p1, camera),
			WorldToGUIPoint(p2, camera)
		};
		Vector2 min = extentPoints[0];
		Vector2 max = extentPoints[0];
		foreach (Vector2 v in extentPoints)
		{
			min = Vector2.Min(min, v);
			max = Vector2.Max(max, v);
		}
		return new Rect(min.x, min.y, max.x - min.x, max.y - min.y);
	}


}
