#!/bin/sh

workdir=bench/
mkdir -p workdir

gym=MountainCarContinuous-v0

out_csv=${workdir}/bench.csv

for n_nodes_per_layer in 64 128 256 512 1024 2700
do
	for n_layers in 2 4 8 16 32
	do
		echo "Layers: ${n_layers} Width: ${n_nodes_per_layer}"
		modelprefix=bench_w${n_nodes_per_layer}xd${n_layers}
		python3 ./train.py ${gym} ${workdir}/${modelprefix} ${n_layers} ${n_nodes_per_layer}
		python3 ./model_conv.py ${gym} ${workdir}/${modelprefix}
	done
done

for modelfile in ${workdir}/*.tflite
do
	echo "Benchmarking ${modelfile}"
	python3 ./tflite_benchmark.py ${gym} ${modelfile} ${out_csv}
done

