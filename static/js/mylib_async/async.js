(function () {
	'use strict';

	var outTextarea = document.getElementById('out');

	function log(text) {
		outTextarea.value += text;
	}


	var optimizer = new pso.Optimizer();

	optimizer.setObjectiveFunction(function (x, done) {
		setTimeout(function () {
			log('x');
			done(-Math.pow(x[0], 2));
		}, Math.random() * 800 + 20);
	}, {
		async: true
	});

	var initialPopulationSize = 20;
	var domain = [new pso.Interval(-5, 5)];

	optimizer.init(initialPopulationSize, domain);

	var iterations = 0, maxIterations = 10;

	function loop() {
		if (iterations >= maxIterations) {
			log([
				'\n--- ---\nOptimasi selesai',
				'Nilai terbaik yang didapatkan f(x): ' + optimizer.getBestFitness(),
				'dengan x = ' + optimizer.getBestPosition()[0]
			].join('\n'));
		} else {
			iterations++;
			log('\nIterasi ke - ' + iterations + ' dari ' + maxIterations + ' ');
			optimizer.step(loop);
		}
	}

	log('Mulai melakukan proses optimasi');
	loop();
})();
