import pdb

def stack_to_string(stack):
	op = ""
	for i in stack:
		if op == "":
			op = op + i
		else:
			op = op + ' ' + i
	return op

def cal_score(outputs, trgs, beam_decode=False):
	corr = 0
	tot = 0
	disp_corr = []
	for i in range(len(outputs)):
		if beam_decode:
			ops = outputs[i]
			cur_corr = []
			for cur_op_idx, op in enumerate(ops):
				tot+=1
				if op == trgs[i]:
					if cur_op_idx == 0:
						corr+=1
					cur_corr.append(1)
				else:
					cur_corr.append(0)
			disp_corr.append(cur_corr)
		else:	
			op = stack_to_string(outputs[i]) 
			if op == trgs[i]:
				corr+=1
				tot+=1
				disp_corr.append(1)
			else:
				tot+=1
				disp_corr.append(0)

	return corr, tot, disp_corr


