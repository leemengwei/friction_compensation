import torch
import torch.distributed as dist
import argparse
from time import sleep
from random import randint

def foo(rank, world_size):
    for step in range(10):
        # get random int
        value = randint(0, 10)
        # group all ranks
        ranks = list(range(world_size))
        group = dist.new_group(ranks=ranks)
        # compute reduced sum
        tensor = torch.IntTensor([value]).to('cuda:0')
        #tensor=torch.cat([tensor,tensor],dim=0)
        dist.all_reduce(tensor, op=dist.ReduceOp.PRODUCT, group=group)
        print('rank: {}, step: {}, value: {}, reduced sum: {}.'.format(rank,step,value,tensor.item()))
        sleep(0.01)

def initialize(rank, world_size, ip, port):
    dist.init_process_group(backend='nccl',init_method='tcp://{}:{}'.format(ip, port),rank=rank,world_size=world_size)
    foo(rank, world_size)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=str, default='20000')
    parser.add_argument('--rank', '-r', type=int)
    parser.add_argument('--world-size', '-s', type=int, default=2)
    args = parser.parse_args()
    print(args)
    initialize(args.rank, args.world_size, args.ip, args.port)

if __name__ == '__main__':
    main()