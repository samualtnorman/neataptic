/* Import */
import { fork } from "child_process"
import { join } from "path"

/*******************************************************************************
                                WEBWORKER
*******************************************************************************/
export class TestWorker {
	constructor(dataSet, cost) {
		this.worker = fork(join(__dirname, "/worker"))

		this.worker.send({ set: dataSet, cost: cost.name })
	}

	evaluate(network) {
		return new Promise((resolve, reject) => {
			var serialized = network.serialize()

			var data = {
				activations: serialized[0],
				states: serialized[1],
				conns: serialized[2]
			}

			var _that = this.worker
			this.worker.on("message", function callback(e) {
				_that.removeListener("message", callback)
				resolve(e)
			})

			this.worker.send(data)
		})
	}

	terminate() {
		this.worker.kill()
	}
}

export default TestWorker
