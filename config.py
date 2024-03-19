import argparse


def load_args():
    parser = argparse.ArgumentParser(description='TEMMA')
    parser.add_argument('--mask_a_length', type=str, default='50')
    parser.add_argument('--mask_b_length', type=str, default='10')
    parser.add_argument('--block_num', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--dropout_mmatten', type=float, default=0.5)
    parser.add_argument('--dropout_mtatten', type=float, default=0.2)
    parser.add_argument('--dropout_ff', type=float, default=0.2)
    parser.add_argument('--dropout_subconnect', type=float, default=0.2)
    parser.add_argument('--dropout_position', type=float, default=0.2)
    parser.add_argument('--dropout_embed', type=float, default=0.2)
    parser.add_argument('--dropout_fc', type=float, default=0.2)
    parser.add_argument('--h', type=int, default=4)
    parser.add_argument('--h_mma', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--modal_num', type=int, default=2)
    parser.add_argument('--embed', type=str, default='temporal')
    parser.add_argument('--levels', type=int, default=5)
    parser.add_argument('--ksize', type=int, default=3)
    parser.add_argument('--ntarget', type=int, default=6)
     

    parser.add_argument('--data_path', type=str, default='D:/Desktop/code/6th-ABAW/dataset/')
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=3e-5) #1e-4
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--n_seeds', type=int, default=5)

    
    args = parser.parse_args()

    

    return args