using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System;

public class ArmController : MonoBehaviour {

	public Transform shoulder;
	public Transform bicep;
	public Transform forearm;

	private HingeJoint shoulderJoint;
	public float shoulderSpeed = 50;

	private HingeJoint bicepJoint;
	private float bicepSpeed = 50;

	private HingeJoint forearmJoint;
	private float forearmSpeed = 50;

	void Start () {
		shoulderJoint = shoulder.GetComponent<HingeJoint> ();
		bicepJoint = bicep.GetComponent<HingeJoint> ();
		forearmJoint = forearm.GetComponent<HingeJoint> ();
	}
	
	// Update is called once per frame
	void Update () {
		
	}


	void FixedUpdate () {
		JointMotor m = shoulderJoint.motor;
		float moveHorizontal = Input.GetAxis ("Horizontal");
		if (moveHorizontal > 0) {
			m.targetVelocity = shoulderSpeed;
		} else if (moveHorizontal < 0) {
			m.targetVelocity = -shoulderSpeed;
		} else {
			m.targetVelocity = 0f;
		}
		shoulderJoint.motor = m;

		m = bicepJoint.motor;
		float moveVertical = Input.GetAxis ("Vertical");
		if (moveVertical > 0) {
			m.targetVelocity = -bicepSpeed;
		} else if (moveVertical < 0) {
			m.targetVelocity = bicepSpeed;
		} else {
			m.targetVelocity = 0f;
		}
		bicepJoint.motor = m;

		m = forearmJoint.motor;
		float moveForearm = Input.GetAxis ("Forearm");
		if (moveForearm > 0) {
			m.targetVelocity = -forearmSpeed;
		} else if (moveForearm < 0) {
			m.targetVelocity = forearmSpeed;
		} else {
			m.targetVelocity = 0f;
		}
		forearmJoint.motor = m;
	}
		
}