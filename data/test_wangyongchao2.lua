
	local joint_tmp = {}
	local joint_end_tmp = {}
	
	joint_curr = GetJointTarget(0)
	joint_tmp = Copy_JOINT_TARGET(joint_curr)
	joint_end_tmp = Copy_JOINT_TARGET(joint_curr)
	
	joint_tmp.robax.rax_1 = -90
	joint_tmp.robax.rax_2 = 0
	joint_tmp.robax.rax_3 = -150 
	joint_tmp.robax.rax_4 = -170
	joint_tmp.robax.rax_5 = -150
	joint_tmp.robax.rax_6 = 0
	
	joint_end_tmp.robax.rax_1 = -90
	joint_end_tmp.robax.rax_2 = 0
	joint_end_tmp.robax.rax_3 = -150 
	joint_end_tmp.robax.rax_4 = 170
	joint_end_tmp.robax.rax_5 = -150
	joint_end_tmp.robax.rax_6 = 0
	
    for i=1,3 do
	  joint_tmp.robax.rax_2 = joint_tmp.robax.rax_2+30
	  joint_end_tmp.robax.rax_2 = joint_end_tmp.robax.rax_2+30
	  	  joint_tmp.robax.rax_3 = -150
	  joint_end_tmp.robax.rax_3  = -150 
	  for j = 1,6 do
	      joint_tmp.robax.rax_3 = joint_tmp.robax.rax_3+30
		  joint_end_tmp.robax.rax_3 = joint_end_tmp.robax.rax_3+30
		  		  joint_tmp.robax.rax_5 = -150
		  joint_end_tmp.robax.rax_5 = -150
		  for k = 1,9 do
	        joint_tmp.robax.rax_5 = joint_tmp.robax.rax_5+30
			joint_end_tmp.robax.rax_5 = joint_end_tmp.robax.rax_5+30

			MoveAbsJ(joint_tmp,spd1,fine,tool0,wobj0,load0)
			MoveAbsJ(joint_end_tmp,spd1,fine,tool0,wobj0,load0)
			MoveAbsJ(joint_tmp,spd1,fine,tool0,wobj0,load0)

			MoveAbsJ(joint_tmp,spd2,fine,tool0,wobj0,load0)
			MoveAbsJ(joint_end_tmp,spd2,fine,tool0,wobj0,load0)
			MoveAbsJ(joint_tmp,spd2,fine,tool0,wobj0,load0)

			MoveAbsJ(joint_tmp,spd3,fine,tool0,wobj0,load0)
			MoveAbsJ(joint_end_tmp,spd3,fine,tool0,wobj0,load0)
			MoveAbsJ(joint_tmp,spd3,fine,tool0,wobj0,load0)

			MoveAbsJ(joint_tmp,spd4,fine,tool0,wobj0,load0)
			MoveAbsJ(joint_end_tmp,spd4,fine,tool0,wobj0,load0)
			MoveAbsJ(joint_tmp,spd4,fine,tool0,wobj0,load0)

			MoveAbsJ(joint_tmp,spd5,fine,tool0,wobj0,load0)
			MoveAbsJ(joint_end_tmp,spd5,fine,tool0,wobj0,load0)
			MoveAbsJ(joint_tmp,spd5,fine,tool0,wobj0,load0)

			MoveAbsJ(joint_tmp,spd6,fine,tool0,wobj0,load0)
			MoveAbsJ(joint_end_tmp,spd6,fine,tool0,wobj0,load0)
			MoveAbsJ(joint_tmp,spd6,fine,tool0,wobj0,load0)

			MoveAbsJ(joint_tmp,spd7,fine,tool0,wobj0,load0)
			MoveAbsJ(joint_end_tmp,spd7,fine,tool0,wobj0,load0)
			MoveAbsJ(joint_tmp,spd7,fine,tool0,wobj0,load0)
			Sleep(200)
			SPEEDDATA("spd7",10000,300,1000,70)
			SPEEDDATA("spd6",10000,240,1000,70)
			SPEEDDATA("spd5",10000,180,1000,70)
			SPEEDDATA("spd4",10000,120,1000,70)
			SPEEDDATA("spd3",10000,60,1000,70)
			SPEEDDATA("spd2",10000,30,1000,70)
			SPEEDDATA("spd1",10000,15,1000,70)
			

		  end 
	  end 
	end
	