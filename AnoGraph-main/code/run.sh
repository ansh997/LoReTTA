make clean
make
echo "Running AnoEdge-G"
./main anoedge_g wikipedia_degree 2 32 0.9

echo "Running AnoEdge-L"
./main anoedge_l wikipedia_degree 2 32 0.9

python3 metrics.py --dataset wikipedia_degree