/*******************************************************************************
                                      RATE
*******************************************************************************/

// https://stackoverflow.com/questions/30033096/what-is-lr-policy-in-caffe/30045244

export function FIXED() {
	var func = function(baseRate, iteration) {
		return baseRate
	}
	return func
}

export function STEP(gamma, stepSize) {
	gamma = gamma || 0.9
	stepSize = stepSize || 100

	var func = function(baseRate, iteration) {
		return baseRate * Math.pow(gamma, Math.floor(iteration / stepSize))
	}

	return func
}

export function EXP(gamma) {
	gamma = gamma || 0.999

	var func = function(baseRate, iteration) {
		return baseRate * Math.pow(gamma, iteration)
	}

	return func
}

export function INV(gamma, power) {
	gamma = gamma || 0.001
	power = power || 2

	var func = function(baseRate, iteration) {
		return baseRate * Math.pow(1 + gamma * iteration, -power)
	}

	return func
}
