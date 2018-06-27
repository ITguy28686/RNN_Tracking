import tensorflow as tf
import tensorflow.contrib.slim as slim

class LSTMcell(object):
    def __init__(self, incoming, D_input, D_cell, initializer,
               f_bias=1.0, L2=False, h_act=tf.tanh,
               init_h=None, init_c=None):
        # 屬性
        self.incoming = incoming # 輸入數據
        self.D_input = D_input
        self.D_cell = D_cell
        self.initializer = initializer # 初始化方法
        self.f_bias = f_bias # 遺忘門的初始偏移量
        self.h_act = h_act # 這裡可以選擇LSTM的hidden state的激活函數
        self.type = 'lstm' # 區分gru
        
        # 如果沒有提供最初的hidden state和memory cell，會全部初始為0
        if init_h is None and init_c is None:
            # If init_h and init_c are not provided, initialize them
            # the shape of init_h and init_c is [n_samples, D_cell]
            self.init_h = tf.matmul(self.incoming[0,:,:], tf.zeros([self.D_input, self.D_cell]))
            self.init_c = self.init_h
            self.previous = tf.stack([self.init_h, self.init_c])
            
        # LSTM所有需要學習的參數
        # 每個都是[W_x, W_h, b_f]的tuple
        self.igate = self.Gate()
        self.fgate = self.Gate(bias = f_bias)
        self.ogate = self.Gate()
        self.cell = self.Gate()
        
        # 因為所有的gate都會乘以當前的輸入和上一時刻的hidden state
        # 將矩陣concat在一起，計算後再逐一分離，加快運行速度
        # W_x的形狀是[D_input, 4*D_cell]
        self.W_x = tf.concat(values=[self.igate[0], self.fgate[0], self.ogate[0], self.cell[0]], axis=1)
        self.W_h = tf.concat(values=[self.igate[1], self.fgate[1], self.ogate[1], self.cell[1]], axis=1)
        self.b = tf.concat(values=[self.igate[2], self.fgate[2], self.ogate[2], self.cell[2]], axis=0)
        
        # 對LSTM的權重進行L2 regularization
        if L2:
            self.L2_loss = tf.nn.l2_loss(self.W_x) + tf.nn.l2_loss(self.W_h)
          
    # 初始化gate的函數
    def Gate(self, bias = 0.001):
        # Since we will use gate multiple times, let's code a class for reusing
        Wx = self.initializer([self.D_input, self.D_cell])
        Wh = self.initializer([self.D_cell, self.D_cell])
        b = tf.Variable(tf.constant(bias, shape=[self.D_cell]),trainable=True)
        return Wx, Wh, b
        
    # 大矩陣乘法運算完畢後，方便用於分離各個gate
    def Slice_W(self, x, n):
        # split W's after computing
        return x[:, n*self.D_cell:(n+1)*self.D_cell]
      
    # 每個time step需要運行的步驟
    def Step(self, previous_h_c_tuple, current_x):
        # 分離上一時刻的hidden state和memory cell
        prev_h, prev_c = tf.unstack(previous_h_c_tuple)
        # 統一在concat成的大矩陣中一次完成所有的gates計算
        gates = tf.matmul(current_x, self.W_x) + tf.matmul(prev_h, self.W_h) + self.b
        # 分離輸入門
        i = tf.sigmoid(self.Slice_W(gates, 0))
        # 分離遺忘門
        f = tf.sigmoid(self.Slice_W(gates, 1))
        # 分離輸出門
        o = tf.sigmoid(self.Slice_W(gates, 2))
        # 分離新的更新信息
        c = tf.tanh(self.Slice_W(gates, 3))
        # 利用gates進行當前memory cell的計算
        current_c = f*prev_c + i*c
        # 利用gates進行當前hidden state的計算
        current_h = o*self.h_act(current_c)
        return tf.stack([current_h, current_c])
        
def RNN(cell, cell_b=None, merge='sum'):
    """
    Note that the input shape should be [n_steps, n_sample, D_output],
    and the output shape will also be [n_steps, n_sample, D_output].
    If the original data has a shape of [n_sample, n_steps, D_input],
    use 'inputs_T = tf.transpose(inputs, perm=[1,0,2])'.
    """

    # forward rnn loop
    hstates = tf.scan(fn = cell.Step,
                    elems = cell.incoming,
                    initializer = cell.previous,
                    name = 'hstates')
    if cell.type == 'lstm':
        hstates = hstates[:,0,:,:]
    # reverse the input sequence
    if cell_b is not None:
        incoming_b = tf.reverse(cell.incoming, axis=[0])
    
    # backward rnn loop
        b_hstates_rev = tf.scan(fn = cell_b.Step,
                    elems = incoming_b,
                    initializer = cell_b.previous,
                    name = 'b_hstates')
        if cell_b.type == 'lstm':
            b_hstates_rev = b_hstates_rev[:,0,:,:]
            
        b_hstates = tf.reverse(b_hstates_rev, axis=[0])
    
        if merge == 'sum':
            hstates = hstates + b_hstates
        else:
            hstates = tf.concat(values=[hstates, b_hstates], axis=2)
    return hstates   