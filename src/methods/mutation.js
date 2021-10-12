/* Import */
import {
	LOGISTIC,
	TANH,
	RELU,
	IDENTITY,
	STEP,
	SOFTSIGN,
	SINUSOID,
	GAUSSIAN,
	BENT_IDENTITY,
	BIPOLAR,
	BIPOLAR_SIGMOID,
	HARD_TANH,
	ABSOLUTE,
	INVERSE,
	SELU
} from "./activation"

/*******************************************************************************
									  MUTATION
*******************************************************************************/

// https://en.wikipedia.org/wiki/mutation_(genetic_algorithm)
export const ADD_NODE = {
	name: "ADD_NODE"
}

export const SUB_NODE = {
	name: "SUB_NODE",
	keep_gates: true
}

export const ADD_CONN = {
	name: "ADD_CONN"
}

export const SUB_CONN = {
	name: "REMOVE_CONN"
}

export const MOD_WEIGHT = {
	name: "MOD_WEIGHT",
	min: -1,
	max: 1
}

export const MOD_BIAS = {
	name: "MOD_BIAS",
	min: -1,
	max: 1
}

export const MOD_ACTIVATION = {
	name: "MOD_ACTIVATION",
	mutateOutput: true,
	allowed: [
		LOGISTIC,
		TANH,
		RELU,
		IDENTITY,
		STEP,
		SOFTSIGN,
		SINUSOID,
		GAUSSIAN,
		BENT_IDENTITY,
		BIPOLAR,
		BIPOLAR_SIGMOID,
		HARD_TANH,
		ABSOLUTE,
		INVERSE,
		SELU
	]
}

export const ADD_SELF_CONN = {
	name: "ADD_SELF_CONN"
}

export const SUB_SELF_CONN = {
	name: "SUB_SELF_CONN"
}

export const ADD_GATE = {
	name: "ADD_GATE"
}

export const SUB_GATE = {
	name: "SUB_GATE"
}

export const ADD_BACK_CONN = {
	name: "ADD_BACK_CONN"
}

export const SUB_BACK_CONN = {
	name: "SUB_BACK_CONN"
}

export const SWAP_NODES = {
	name: "SWAP_NODES",
	mutateOutput: true
}

export const ALL = [
	ADD_NODE,
	SUB_NODE,
	ADD_CONN,
	SUB_CONN,
	MOD_WEIGHT,
	MOD_BIAS,
	MOD_ACTIVATION,
	ADD_GATE,
	SUB_GATE,
	ADD_SELF_CONN,
	SUB_SELF_CONN,
	ADD_BACK_CONN,
	SUB_BACK_CONN,
	SWAP_NODES
]

export const FFW = [ ADD_NODE, SUB_NODE, ADD_CONN, SUB_CONN, MOD_WEIGHT, MOD_BIAS, MOD_ACTIVATION, SWAP_NODES ]
