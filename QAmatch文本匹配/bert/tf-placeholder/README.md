# ǰ��
>>�ð汾������tf1.x������Ϊtf.placeholder,ʵ�ֵ�bert,albert�Լ�robert��qa�ı�ƥ�䡣
���ݸ�ʽΪ��query reply label Ҳ�������仰+label,����Ԥ��reply�Ƿ��Ӧquery��seq_id���п���Ϊ�κ�ֵ��������û�в���ѵ����
    
# ���л���    
    ���requirements.txt
# ����ִ��
    ע���ɵ���ֻ����tf-placeholderĿ¼�µ��ļ����Ϳ���ִ��
    1������bert,����main.py
# ��Ҫע��ģ�
>> 1������version���л�bert�İ汾��Ŀǰ֧��bert_modelĿ¼�¼���bert�汾����Ҫ��������汾���ɲο���ӡ�����github�Ĵ�С���ƣ����а汾��bertģ�������аٶ��Ƶ�ַ���ɴӰٶ������أ������ö�Ӧλ��  
>> 2����mode��='train',�������Ԥ�⡣��fold_num>1�����fold_num���ں�  
>> 3��ÿ100��steps��ӡһ��ѧϰ�ʺ�ѵ��loss,ÿ��epoch��ӡһ��ѵ������֤f1score,���restore_on_train=True
��������֤���÷���ߵ�ģ�ͣ�����ÿ�μ�������ģ�ͼ���ѵ�������restore_on_train=False,������ѵ����ÿ��epoch����һ��ģ�͡�
    
