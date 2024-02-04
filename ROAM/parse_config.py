import configparser
import time

class Args:
    def __init__(self):
        pass

def parse_args(config_dir, split_seed):
    args = Args 
    cf = configparser.ConfigParser()
    cf.read(config_dir)
    task = cf.sections()[0]
    print(f'Current task: {task}, Configuration file: {config_dir}')

    args.task = task
    
    ####
    # random seed of the model for training
    args.seed = cf.getint(task, 'seed')
    # random seed for spliting training data (s0~s4)
    args.split_seed = split_seed
    # stage of the process, train or test
    args.stage = cf.get(task, 'stage')
    # pre-trained model for extracting patch features: ImageNet(default), RetCCL, ctraspath, simclr-ciga
    args.embed_type = cf.get(task, 'embed_type')
    # percentage of data used for training: 20%,40%,60%,100%(default)
    args.sample_size = cf.getint(task, 'sample_size')
    # whether to use stain normalization: True(default) or False
    args.not_stainnorm = cf.getboolean(task, 'not_stainnorm')
    # type of test dataset: xiangya(in-house validation), TCGA(external validation)
    args.test_dataset = cf.get(task, 'test_dataset')
    # root directory of feature data
    args.data_root_dir = cf.get(task,'data_root_dir')
    # root directory to save all results
    args.results_dir = cf.get(task, 'results_dir')

    #### training
    # the maxmium epochs of training epochs allowed
    args.max_epochs = cf.getint(task, 'max_epochs')
    # size of a training batch
    args.batch_size = cf.getint(task, 'batch_size')
    args.lr = cf.getfloat(task, 'lr')
    args.optimizer = cf.get(task, 'optimizer')
    args.weight_decay = cf.getfloat(task, 'weight_decay')
    args.scheduler = cf.get(task, 'scheduler')
    # early stop when metrics have not improved for certain epochs
    args.stop_epochs = cf.getint(task, 'stop_epochs')
    # sampling weights for each class in the dataloader, for class imbalance
    args.weighted_sample = cf.getboolean(task, 'weighted_sample')
    args.emb_dropout = cf.getfloat(task, 'emb_dropout')
    args.attn_dropout = cf.getfloat(task, 'attn_dropout')
    args.dropout = cf.getfloat(task, 'dropout')

    #### ROAM specific options
    # name of the model
    args.model_type = cf.get(task, 'model_type')
    args.roi_dropout = cf.getboolean(task, 'roi_dropout')
    args.roi_supervise = cf.getboolean(task, 'roi_supervise')
    args.roi_weight = cf.getfloat(task, 'roi_weight')
    # the number of instances used in instance-level supervision
    args.topk = cf.getint(task, 'topk')
    # size of ROI at 20x, (0:2048,1:1024,2:512)
    args.roi_level = cf.getint(task, 'roi_level')
    # multi-scale ('ms') or single scale ('ss')
    args.scale_type = cf.get(task, 'scale_type')
    # magnification scale of input ROI for single-scale model. (0:20x,1:10x,2:5x)
    args.single_level = cf.getint(task, 'single_level')
    # weight coefficient of instance embedding at each magnificant level
    args.embed_weightx5 = eval(cf.get(task, 'embed_weightx5'))
    args.embed_weightx10 = eval(cf.get(task, 'embed_weightx10'))
    args.embed_weightx20 = eval(cf.get(task, 'embed_weightx20'))
    # whether to use inter-scale self-attention module. False (with inter-scale SA), True (without inter-scale SA)
    args.not_interscale = cf.getboolean(task, 'not_interscale')
    # model config
    args.dim = cf.getint(task, 'dim')
    args.depths = eval(cf.get(task, 'depths'))
    args.heads = cf.getint(task, 'heads')
    args.mlp_dim = cf.getint(task, 'mlp_dim')
    args.dim_head = cf.getint(task, 'dim_head')
    args.pool = cf.get(task, 'pool')
    args.ape = cf.getboolean(task, 'ape')
    args.attn_type = cf.get(task, 'attn_type')
    args.shared_pe = cf.getboolean(task, 'shared_pe')
    # name of the experiment
    args.exp_code = '_'.join(map(str, [args.task, args.depths,
                                       args.embed_type,
                                       args.batch_size, args.roi_dropout, 
                                       args.roi_supervise,
                                       args.roi_weight, args.topk,
                                       args.roi_level,
                                       args.scale_type, args.single_level,
                                       args.not_interscale]))

    print('exp_code: {}'.format(args.exp_code))
    # information of all tasks
    # example tasks
    task_info = {
        ## in-house validataion with xiangya test dataset
        'int_glioma_detection':{'csv_path':'../data_prepare/data_csv/xiangya_data_info_pro.csv',
                          'label_dict':{0:0,1:1,2:2,3:2,4:2},
                          'n_classes': 3,
                          'split_dir': f'../data_prepare/data_split/xiangya_split_detection/xiangya_split_detection_{split_seed}.npy',
                          'test_split_dir': '../data_prepare/data_split/xiangya_split_detection/test_split_label_detection.npy',
                          'cls_weights':[50,24,514]},

        'int_glioma_tumor_subtyping':{'csv_path': '../data_prepare/data_csv/example_xiangya_data_info_pro.csv',
                                      'label_dict': {i+2:i for i in range(3)},
                                      'n_classes': 3,
                                      'split_dir': f'../data_prepare/data_split/xiangya_split_subtype/xiangya_split_subtype_size{args.sample_size}_{split_seed}.npy',
                                      'test_split_dir': '../data_prepare/data_split/xiangya_split_subtype/example_test_split.npy',
                                      'cls_weights':[281,119,111]},
        # external validation with TCGA test dataset
        'ext_glioma_tumor_subtyping3':{'csv_path':'../data_prepare/data_csv/tcga_data_info_pro.csv',
                                       'label_dict':{3:0,4:0,5:0,7:1,8:1},
                                       'n_classes': 2,
                                       'split_dir': f'../data_prepare/data_split/xiangya_tcga_split_subtype2/xiangya_split_subtype2_{split_seed}.npy',
                                       'label_dict_ext': {'astrocytoma_G2':0,'astrocytoma_G3':0,'glioblastoma_G4':0,'oligodendroglioma_G2':1,'oligodendroglioma_G3':1},
                                       'test_split_dir_ext': '../data_prepare/data_split/xiangya_tcga_split_subtype2/TCGA_test_split_label_subtype2.npy',
                                       'cls_weights':[261,179]},

    }
    
    
    return args, task_info


def parse_args_heatmap_roi(config_dir, split_seed):
    args = Args
    cf = configparser.ConfigParser()
    cf.read(config_dir)
    task = cf.sections()[0]
    print(f'Current task: {task}, Configuration file: {config_dir}')

    args.task = task
    
    ####
    args.seed = cf.getint(task, 'seed')
    args.split_seed = split_seed
    args.embed_type = cf.get(task, 'embed_type')
    args.not_stainnorm = cf.getboolean(task, 'not_stainnorm')
    args.emb_dropout = cf.getfloat(task, 'emb_dropout')
    args.attn_dropout = cf.getfloat(task, 'attn_dropout')
    args.dropout = cf.getfloat(task, 'dropout')

    args.batch_size = cf.getint(task, 'batch_size')
    
    #### ROAM specific options
    args.model_type = cf.get(task, 'model_type')
    args.roi_dropout = cf.getboolean(task, 'roi_dropout')
    args.roi_supervise = cf.getboolean(task, 'roi_supervise')
    args.roi_weight = cf.getfloat(task, 'roi_weight')
    args.topk = cf.getint(task, 'topk')
    args.roi_level = cf.getint(task, 'roi_level')
    args.scale_type = cf.get(task, 'scale_type')
    args.single_level = cf.getint(task, 'single_level')
    args.embed_weightx5 = eval(cf.get(task, 'embed_weightx5'))
    args.embed_weightx10 = eval(cf.get(task, 'embed_weightx10'))
    args.embed_weightx20 = eval(cf.get(task, 'embed_weightx20'))
    args.not_interscale = cf.getboolean(task, 'not_interscale')

    args.dim = cf.getint(task, 'dim')
    args.depths = eval(cf.get(task, 'depths'))
    args.heads = cf.getint(task, 'heads')
    args.mlp_dim = cf.getint(task, 'mlp_dim')
    args.dim_head = cf.getint(task, 'dim_head')
    args.pool = cf.get(task, 'pool')
    args.ape = cf.getboolean(task, 'ape')
    args.attn_type = cf.get(task, 'attn_type')
    args.shared_pe = cf.getboolean(task, 'shared_pe')

    # roi vis parameters
    #args.image_path = cf.get(task,'image_path')
    args.process_list = cf.get(task,'process_list')
    args.topk_num = cf.getint(task,'topk_num')
    args.vis_type = cf.get(task,'vis_type')
    args.vis_scale = cf.get(task,'vis_scale')
    args.sample = cf.get(task,'sample')
    args.level = cf.getint(task, 'level')
    args.head_fusion = cf.get(task, 'head_fusion')
    args.discard_ratio = cf.getfloat(task, 'discard_ratio')
    args.category_index = cf.getint(task,'category_index')

    args.exp_code = '_'.join(map(str, [args.task, args.depths,
                                       args.embed_type,
                                       args.batch_size, args.roi_dropout, 
                                       args.roi_supervise,
                                       args.roi_weight, args.topk,
                                       args.roi_level,
                                       args.scale_type,args.single_level,
                                       args.not_interscale]))

    task_info = {
        ## in-house validataion with xiangya test dataset
        'int_glioma_detection':{'csv_path':'../data_prepare/data_csv/xiangya_data_info_pro.csv',
                          'label_dict':{0:0,1:1,2:2,3:2,4:2},
                          'n_classes': 3,
                          'split_dir': f'../data_prepare/data_split/xiangya_split_detection/xiangya_split_detection_{split_seed}.npy',
                          'test_split_dir': '../data_prepare/data_split/xiangya_split_detection/test_split_label_detection.npy',
                          'cls_weights':[50,24,514]},

        'int_glioma_tumor_subtyping':{'csv_path': '../data_prepare/data_csv/example_xiangya_data_info_pro.csv',
                                      'label_dict': {i+2:i for i in range(3)},
                                      'n_classes': 3,
                                      'split_dir': f'../data_prepare/data_split/xiangya_split_subtype/xiangya_split_subtype_size100_{split_seed}.npy',
                                      'test_split_dir': '../data_prepare/data_split/xiangya_split_subtype/example_test_split.npy',
                                      'cls_weights':[281,119,111]},
        # external validation with TCGA test dataset
        'ext_glioma_tumor_subtyping3':{'csv_path':'../data_prepare/data_csv/tcga_data_info_pro.csv',
                                       'label_dict':{3:0,4:0,5:0,7:1,8:1},
                                       'n_classes': 2,
                                       'split_dir': f'../data_prepare/data_split/xiangya_tcga_split_subtype2/xiangya_split_subtype2_{split_seed}.npy',
                                       'label_dict_ext': {'astrocytoma_G2':0,'astrocytoma_G3':0,'glioblastoma_G4':0,'oligodendroglioma_G2':1,'oligodendroglioma_G3':1},
                                       'test_split_dir_ext': '../data_prepare/data_split/xiangya_tcga_split_subtype2/TCGA_test_split_label_subtype2.npy',
                                       'cls_weights':[261,179]},


    }
    return args, task_info


def read_taskinfo(split_seed): 
    
    task_info = {
        ## in-house validataion with xiangya test dataset
        'int_glioma_detection':{'csv_path':'../data_prepare/data_csv/xiangya_data_info_pro.csv',
                          'label_dict':{0:0,1:1,2:2,3:2,4:2},
                          'n_classes': 3,
                          'split_dir': f'../data_prepare/data_split/xiangya_split_detection/xiangya_split_detection_{split_seed}.npy',
                          'test_split_dir': '../data_prepare/data_split/xiangya_split_detection/test_split_label_detection.npy',
                          'cls_weights':[50,24,514]},

        'int_glioma_tumor_subtyping':{'csv_path': '../data_prepare/data_csv/xiangya_data_info_pro.csv',
                                      'label_dict': {i+2:i for i in range(3)},
                                      'n_classes': 3,
                                      'split_dir': f'../data_prepare/data_split/xiangya_split_subtype/xiangya_split_subtype_size100_{split_seed}.npy',
                                      'test_split_dir': '../data_prepare/data_split/xiangya_split_subtype/test_split_label_subtype.npy',
                                      'cls_weights':[281,119,111]},
        # external validation with TCGA test dataset
        'ext_glioma_tumor_subtyping3':{'csv_path':'../data_prepare/data_csv/tcga_data_info_pro.csv',
                                       'label_dict':{3:0,4:0,5:0,7:1,8:1},
                                       'n_classes': 2,
                                       'split_dir': f'../data_prepare/data_split/xiangya_tcga_split_subtype2/xiangya_split_subtype2_{split_seed}.npy',
                                       'label_dict_ext': {'astrocytoma_G2':0,'astrocytoma_G3':0,'glioblastoma_G4':0,'oligodendroglioma_G2':1,'oligodendroglioma_G3':1},
                                       'test_split_dir_ext': '../data_prepare/data_split/xiangya_tcga_split_subtype2/TCGA_test_split_label_subtype2.npy',
                                       'cls_weights':[261,179]},

    }
    return task_info


