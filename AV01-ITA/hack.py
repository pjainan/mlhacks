import dataprocessing as dp

def ProcessData():
    dh = dp.DataHelper()
    dh.fill_arrays('test')
    dh.fill_arrays('train')
    dh.get_data('train')
    dh.get_data('test')
    dh.show_pickle()
    
if __name__ == '__main__':
    ProcessData()