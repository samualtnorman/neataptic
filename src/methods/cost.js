/*******************************************************************************
                                    COST FUNCTIONS
*******************************************************************************/

// https://en.wikipedia.org/wiki/Loss_function

// Cross entropy error
export function CROSS_ENTROPY(target, output) {
	var error = 0
	for (var i = 0; i < output.length; i++) {
		// Avoid negative and zero numbers, use 1e-15 http://bit.ly/2p5W29A
		error -=
			target[i] * Math.log(Math.max(output[i], 1e-15)) +
			(1 - target[i]) * Math.log(1 - Math.max(output[i], 1e-15))
	}
	return error / output.length
}

// Mean Squared Error
export function MSE(target, output) {
	var error = 0
	for (var i = 0; i < output.length; i++) {
		error += Math.pow(target[i] - output[i], 2)
	}

	return error / output.length
}

// Binary error
export function BINARY(target, output) {
	var misses = 0
	for (var i = 0; i < output.length; i++) {
		misses += Math.round(target[i] * 2) !== Math.round(output[i] * 2)
	}

	return misses
}

// Mean Absolute Error
export function MAE(target, output) {
	var error = 0
	for (var i = 0; i < output.length; i++) {
		error += Math.abs(target[i] - output[i])
	}

	return error / output.length
}

// Mean Absolute Percentage Error
export function MAPE(target, output) {
	var error = 0
	for (var i = 0; i < output.length; i++) {
		error += Math.abs((output[i] - target[i]) / Math.max(target[i], 1e-15))
	}

	return error / output.length
}

// Mean Squared Logarithmic Error
export function MSLE(target, output) {
	var error = 0
	for (var i = 0; i < output.length; i++) {
		error += Math.log(Math.max(target[i], 1e-15)) - Math.log(Math.max(output[i], 1e-15))
	}

	return error
}

// Hinge loss, for classifiers
export function HINGE(target, output) {
	var error = 0
	for (var i = 0; i < output.length; i++) {
		error += Math.max(0, 1 - target[i] * output[i])
	}

	return error
}
