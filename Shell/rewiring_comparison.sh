# SDRF
python main.py --cfg configs/VANILLA/COLLAB/COLLAB-GCN.yaml --hops 10 --split 25 --rewiring sdrf
python main.py --cfg configs/VANILLA/IMDB-BINARY/IMDB-BINARY-GCN.yaml --hops 10 --split 25 --rewiring sdrf
python main.py --cfg configs/VANILLA/REDDIT-BINARY/REDDIT-BINARY-GCN.yaml --hops 10 --split 25 --rewiring sdrf

# FoSR
python main.py --cfg configs/VANILLA/IMDB-BINARY/IMDB-BINARY-GCN.yaml --hops 10 --split 25 --rewiring fosr 
python main.py --cfg configs/VANILLA/REDDIT-BINARY/REDDIT-BINARY-GCN.yaml --hops 10 --split 25 --rewiring fosr 
