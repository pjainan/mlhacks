import dataprocessing as dp

def ProcessData():
    dh = dp.DataHelper()
    dh.get_data('train')
    dh.get_data('test')
    dh.fill_arrays('test')
    dh.fill_arrays('train')
    dh.pickle_dump()
    dh.pickle_extract()

if __name__ == '__main__':
    ProcessData()