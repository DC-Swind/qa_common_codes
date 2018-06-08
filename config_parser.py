    main_start_time = time.strftime("%d %b %Y %H:%M:%S", time.localtime())

    config = Config()
    parser = argparse.ArgumentParser(description='resAttention-ver10')
    parser.add_argument('--save_name', type=str, default='resAtt_ver10_new_test', help='filename for save model and log.')
    parser.add_argument('--cuda', type=str, default='0', help='gpu id')
    parser.add_argument('--end_time_delay', type=int, default=1, help='time decay after ending.')
    parser.add_argument('--residual_pooling', dest='residual_pooling', action='store_true')
    parser.add_argument('--keep_weight_prob', type=float, default=1.0, help='drop rate of attention weight.')
    parser.add_argument('--keep_residual_prob', type=float, default=1.0, help='drop rate of residual connection.')
    parser.add_argument('--keep_input_prob', type=float, default=0.7, help='drop rate of word embeddings.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--training_eps', type=int, default=30, help='max epochs for training.')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size.')
    parser.add_argument('--local_window_size', type=int, default=11, help='attention window size.')
    parser.add_argument('--n_input', type=int, default=300, help='word embedding size.')
    parser.add_argument('--n_steps', type=int, default=50, help='sequence length.')
    parser.add_argument('--n_hidden', type=int, default=300, help='hidden size.')
    parser.add_argument('--embedding_tuning', dest='embedding_tuning', action='store_true')
    parser.add_argument('--opt', type=str, default="sgd", help='optimizer')
    parser.add_argument('--alpha', type=float, default=0.99, help='alpha of rmsprop')
    parser.add_argument('--using_LN', dest='using_LN', action='store_true')
    args = parser.parse_args()
    
    config.save_name = args.save_name
    config.restore_name = args.save_name
    config.save_path = config.path+"models/"+config.save_name # model path to restore
    config.restore_path = config.path+"models/"+config.restore_name
    config.cuda = args.cuda   
    config.end_time_delay = args.end_time_delay
    config.residual_pooling = args.residual_pooling
    config.keep_weight_prob = args.keep_weight_prob
    config.keep_residual_prob = args.keep_residual_prob
    config.keep_input_prob = args.keep_input_prob
    config.learning_rate = args.learning_rate
    config.lr_init = args.learning_rate
    config.training_eps = args.training_eps # training epoches
    config.batch_size = args.batch_size
    config.local_window_size = args.local_window_size
    config.n_input = args.n_input # word embedding
    config.n_steps = args.n_steps # timesteps, sequence max len, over that will be discarded.
    config.n_hidden = args.n_hidden # hidden layer num of features
    config.embedding_tuning = args.embedding_tuning
    config.opt = args.opt
    config.alpha = args.alpha
    config.using_LN = args.using_LN
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda
