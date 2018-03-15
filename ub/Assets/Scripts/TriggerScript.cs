using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TriggerScript : MonoBehaviour {
	void OnTriggerEnter(Collider other) {
		if (other.CompareTag ("berry")) {
			Object a = other.gameObject.transform.parent;
			BerriesController c = other.gameObject.transform.parent.GetComponent<BerriesController> ();
			c.CollectBerry (other.gameObject.GetComponent<Transform>());
		}
	}
}
