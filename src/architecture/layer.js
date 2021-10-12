import { activation as _activation, connection, gating } from "../methods"
import Group from "./group"
import Node from "./Node"

/*******************************************************************************
										 Group
*******************************************************************************/
export class Layer {
	constructor() {
		this.output = null

		this.nodes = []
		this.connections = {
			in: [],
			out: [],
			self: []
		}
	}

	/**
	 * Activates all the nodes in the group
	 */
	activate(value) {
		var values = []

		if (typeof value !== "undefined" && value.length !== this.nodes.length) {
			throw new Error("Array with values should be same as the amount of nodes!")
		}

		for (var i = 0; i < this.nodes.length; i++) {
			var activation
			if (typeof value === "undefined") {
				activation = this.nodes[i].activate()
			} else {
				activation = this.nodes[i].activate(value[i])
			}

			values.push(activation)
		}

		return values
	}

	/**
	 * Propagates all the node in the group
	 */
	propagate(rate, momentum, target) {
		if (typeof target !== "undefined" && target.length !== this.nodes.length) {
			throw new Error("Array with values should be same as the amount of nodes!")
		}

		for (var i = this.nodes.length - 1; i >= 0; i--) {
			if (typeof target === "undefined") {
				this.nodes[i].propagate(rate, momentum, true)
			} else {
				this.nodes[i].propagate(rate, momentum, true, target[i])
			}
		}
	}

	/**
	 * Connects the nodes in this group to nodes in another group or just a node
	 */
	connect(target, method, weight) {
		var connections
		if (target instanceof Group || target instanceof Node) {
			connections = this.output.connect(target, method, weight)
		} else if (target instanceof Layer) {
			connections = target.input(this, method, weight)
		}

		return connections
	}

	/**
	 * Make nodes from this group gate the given connection(s)
	 */
	gate(connections, method) {
		this.output.gate(connections, method)
	}

	/**
	 * Sets the value of a property for every node
	 */
	set(values) {
		for (var i = 0; i < this.nodes.length; i++) {
			var node = this.nodes[i]

			if (node instanceof Node) {
				if (typeof values.bias !== "undefined") {
					node.bias = values.bias
				}

				node.squash = values.squash || node.squash
				node.type = values.type || node.type
			} else if (node instanceof Group) {
				node.set(values)
			}
		}
	}

	/**
	 * Disconnects all nodes from this group from another given group/node
	 */
	disconnect(target, twosided) {
		twosided = twosided || false

		// In the future, disconnect will return a connection so indexOf can be used
		var i, j, k
		if (target instanceof Group) {
			for (i = 0; i < this.nodes.length; i++) {
				for (j = 0; j < target.nodes.length; j++) {
					this.nodes[i].disconnect(target.nodes[j], twosided)

					for (k = this.connections.out.length - 1; k >= 0; k--) {
						let conn = this.connections.out[k]

						if (conn.from === this.nodes[i] && conn.to === target.nodes[j]) {
							this.connections.out.splice(k, 1)
							break
						}
					}

					if (twosided) {
						for (k = this.connections.in.length - 1; k >= 0; k--) {
							let conn = this.connections.in[k]

							if (conn.from === target.nodes[j] && conn.to === this.nodes[i]) {
								this.connections.in.splice(k, 1)
								break
							}
						}
					}
				}
			}
		} else if (target instanceof Node) {
			for (i = 0; i < this.nodes.length; i++) {
				this.nodes[i].disconnect(target, twosided)

				for (j = this.connections.out.length - 1; j >= 0; j--) {
					let conn = this.connections.out[j]

					if (conn.from === this.nodes[i] && conn.to === target) {
						this.connections.out.splice(j, 1)
						break
					}
				}

				if (twosided) {
					for (k = this.connections.in.length - 1; k >= 0; k--) {
						let conn = this.connections.in[k]

						if (conn.from === target && conn.to === this.nodes[i]) {
							this.connections.in.splice(k, 1)
							break
						}
					}
				}
			}
		}
	}

	/**
	 * Clear the context of this group
	 */
	clear() {
		for (var i = 0; i < this.nodes.length; i++) {
			this.nodes[i].clear()
		}
	}
}

export class Dense extends Layer {
	constructor(size) {
		super()

		// Init required nodes (in activation order)
		var block = new Group(size)

		this.nodes.push(block)
		this.output = block

		this.input = function(from, method, weight) {
			if (from instanceof Layer) from = from.output
			method = method || connection.ALL_TO_ALL
			return from.connect(block, method, weight)
		}
	}
}

export class LSTM extends Layer {
	constructor(size) {
		super()

		// Init required nodes (in activation order)
		var inputGate = new Group(size)
		var forgetGate = new Group(size)
		var memoryCell = new Group(size)
		var outputGate = new Group(size)
		var outputBlock = new Group(size)

		inputGate.set({
			bias: 1
		})
		forgetGate.set({
			bias: 1
		})
		outputGate.set({
			bias: 1
		})

		// Set up internal connections
		memoryCell.connect(inputGate, connection.ALL_TO_ALL)
		memoryCell.connect(forgetGate, connection.ALL_TO_ALL)
		memoryCell.connect(outputGate, connection.ALL_TO_ALL)
		var forget = memoryCell.connect(memoryCell, connection.ONE_TO_ONE)
		var output = memoryCell.connect(outputBlock, connection.ALL_TO_ALL)

		// Set up gates
		forgetGate.gate(forget, gating.SELF)
		outputGate.gate(output, gating.OUTPUT)

		// Add to nodes array
		this.nodes = [ inputGate, forgetGate, memoryCell, outputGate, outputBlock ]

		// Define output
		this.output = outputBlock

		this.input = function(from, method, weight) {
			if (from instanceof Layer) from = from.output
			method = method || connection.ALL_TO_ALL
			var connections = []

			var input = from.connect(memoryCell, method, weight)
			connections = connections.concat(input)

			connections = connections.concat(from.connect(inputGate, method, weight))
			connections = connections.concat(from.connect(outputGate, method, weight))
			connections = connections.concat(from.connect(forgetGate, method, weight))

			inputGate.gate(input, gating.INPUT)

			return connections
		}
	}
}

export class GRU extends Layer {
	constructor(size) {
		super()

		var updateGate = new Group(size)
		var inverseUpdateGate = new Group(size)
		var resetGate = new Group(size)
		var memoryCell = new Group(size)
		var output = new Group(size)
		var previousOutput = new Group(size)

		previousOutput.set({
			bias: 0,
			squash: _activation.IDENTITY,
			type: "constant"
		})
		memoryCell.set({
			squash: _activation.TANH
		})
		inverseUpdateGate.set({
			bias: 0,
			squash: _activation.INVERSE,
			type: "constant"
		})
		updateGate.set({
			bias: 1
		})
		resetGate.set({
			bias: 0
		})

		// Update gate calculation
		previousOutput.connect(updateGate, connection.ALL_TO_ALL)

		// Inverse update gate calculation
		updateGate.connect(inverseUpdateGate, connection.ONE_TO_ONE, 1)

		// Reset gate calculation
		previousOutput.connect(resetGate, connection.ALL_TO_ALL)

		// Memory calculation
		var reset = previousOutput.connect(memoryCell, connection.ALL_TO_ALL)

		resetGate.gate(reset, gating.OUTPUT) // gate

		// Output calculation
		var update1 = previousOutput.connect(output, connection.ALL_TO_ALL)
		var update2 = memoryCell.connect(output, connection.ALL_TO_ALL)

		updateGate.gate(update1, gating.OUTPUT)
		inverseUpdateGate.gate(update2, gating.OUTPUT)

		// Previous output calculation
		output.connect(previousOutput, connection.ONE_TO_ONE, 1)

		// Add to nodes array
		this.nodes = [ updateGate, inverseUpdateGate, resetGate, memoryCell, output, previousOutput ]

		this.output = output

		this.input = function(from, method, weight) {
			if (from instanceof Layer) from = from.output
			method = method || connection.ALL_TO_ALL
			var connections = []

			connections = connections.concat(from.connect(updateGate, method, weight))
			connections = connections.concat(from.connect(resetGate, method, weight))
			connections = connections.concat(from.connect(memoryCell, method, weight))

			return connections
		}
	}
}

export class Memory extends Layer {
	constructor(size, memory) {
		super()
		// Because the output can only be one group, we have to put the nodes all in óne group
		var previous = null
		var i
		for (i = 0; i < memory; i++) {
			var block = new Group(size)

			block.set({
				squash: _activation.IDENTITY,
				bias: 0,
				type: "constant"
			})

			if (previous != null) {
				previous.connect(block, connection.ONE_TO_ONE, 1)
			}

			this.nodes.push(block)
			previous = block
		}

		this.nodes.reverse()

		for (i = 0; i < this.nodes.length; i++) {
			this.nodes[i].nodes.reverse()
		}

		// Because output can only be óne group, fit all memory nodes in óne group
		var outputGroup = new Group(0)
		for (var group in this.nodes) {
			outputGroup.nodes = outputGroup.nodes.concat(this.nodes[group].nodes)
		}
		this.output = outputGroup

		this.input = function(from, method, weight) {
			if (from instanceof Layer) from = from.output
			method = method || connection.ALL_TO_ALL

			if (from.nodes.length !== this.nodes[this.nodes.length - 1].nodes.length) {
				throw new Error("Previous layer size must be same as memory size")
			}

			return from.connect(this.nodes[this.nodes.length - 1], connection.ONE_TO_ONE, 1)
		}
	}
}
