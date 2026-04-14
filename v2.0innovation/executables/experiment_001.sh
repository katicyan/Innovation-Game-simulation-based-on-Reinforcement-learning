python3 ../train_pettingzoo_independent_q.py --episodes 200 --max-steps 100 --n-agents 10 --output-file e001 --seed 0

python3 ../train_pettingzoo_independent_q.py --episodes 200 --max-steps 100 --n-agents 10  --initial-capital [100,50,50,50,50,50,50,50,50,50] --output-file e002 --seed 0

python3 ../train_pettingzoo_independent_q.py --episodes 200 --max-steps 100 --n-agents 10 --initial-technology [3,0,0,0,0,0,0,0,0,0] --output-file e003 --seed 0

python3 ../train_pettingzoo_independent_q.py --episodes 200 --max-steps 100 --n-agents 10 --tech-levels [80,40,20,10,5,1] --output-file e004 --seed 0
