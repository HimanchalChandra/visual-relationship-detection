import argparse


def parse_opts():
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", type=int, default=100,
						help="number of epochs")
	parser.add_argument("--batch_size", type=int, default=32,
						help="size of each image batch")
	parser.add_argument(
		'--learning_rate',
		default=0.1,
		type=float,
		help=
		'Initial learning rate (divided by 10 while training by lr scheduler)')
	parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
	parser.add_argument(
		'--dampening', default=0.9, type=float, help='dampening of SGD')
	parser.add_argument(
		'--weight_decay', default=1e-3, type=float, help='Weight Decay')
	parser.add_argument(
		'--nesterov', action='store_true', help='Nesterov momentum')
	parser.set_defaults(nesterov=False)
	parser.add_argument("--n_cpu", type=int, default=8,
						help="number of cpu threads to use during batch generation")
	parser.add_argument("--img_size", type=int, default=32,
						help="size of each image dimension")
	parser.add_argument("--checkpoint_interval", type=int,
						default=1, help="interval between saving model weights")
	parser.add_argument("--log_interval", type=int, default=10,
						help="interval of display metrics")
	parser.add_argument("--dataset_path", type=str,
						default="/home/neuroplex/data/vrd", help="dataset dir")
	parser.add_argument("--weights", type=str, 
						help="starts from checkpoint model")
	parser.add_argument("--resume_path", type=str,
						default=None, help="resume training")
	parser.add_argument("--save_interval", type=int,
						default=10, help="saving weights interval")
	parser.add_argument("--glove_path", type=str,
						default="/home/neuroplex/glove.6B/glove.6B.300d.txt", help="Path to glove word embeddings")
	parser.add_argument("--dataset", type=str,
						default="vrd", help="Dataset name")
	parser.add_argument(
		'--lr_patience',
		default=10,
		type=int,
		help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
	parser.add_argument("--gpu", type=int,
						default=0, help="gpu id")
	opt = parser.parse_args()
	return opt
