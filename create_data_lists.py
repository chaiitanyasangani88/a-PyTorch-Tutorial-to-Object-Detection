from utils import create_data_lists, create_data_lists_oxford_dataset

if __name__ == '__main__':
    # create_data_lists(voc07_path='/media/ssd/ssd data/VOC2007',
    #                   voc12_path='/media/ssd/ssd data/VOC2012',
    #                   output_folder='./')

    create_data_lists_oxford_dataset('oxford_dataset', './')
