# 使用 smtplib 模块发送纯文本邮件
import smtplib
import ssl
from email.message import EmailMessage

'''
以网易邮箱为例：

准备工作：

1）首先进入网页版网易邮箱，点击设置

2）点击，POP3/SMTP/IMAP

3）开启  IMAP/SMTP服务 和 POP3/SMTP服务  并且记住授权码（一定要记住只会出现一次）
'''

EMAIL_ADDRESS = "souhu3326@163.com"  # 邮箱的地址
EMAIL_PASSWORD = "PMGCAMLQPVFJMJYYXX"  # 授权码

# 连接到smtp服务器
# smtp = smtplib.SMTP('smtp.163.com', 25)     # 未加密

# 也可以使用ssl模块的context加载系统允许的证书，在登录时进行验证
context = ssl.create_default_context()

'''
# 为了防止信息在网络传输中泄漏，最好对信息进行加密
smtp = smtplib.SMTP_SSL("smtp.163.com", 465, context=context)       # 完成加密通讯
# 连接成功后使用login方法登录自己的邮箱
smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
# 一、使用 sendmail 方法实现发送邮件信息
sender = EMAIL_ADDRESS      # 用来发送的邮箱地址
receive = ""        # 目标邮箱地址
subject = "邮件标题内容"
body = "邮件主体内容"
msg = f"Subject: {subject}\n\n{body}"
smtp.sendmail(sender, receive, msg)
# 发送完毕后使用 quit 方法关闭连接
smtp.quit()
'''

def send_msg(subject,content):
    subject = subject
    body = content

    msg = EmailMessage()
    msg['subject'] = subject  # 邮件标题
    msg['From'] = EMAIL_ADDRESS  # 邮件发件人
    msg['To'] = "ttjjlw123@163.com"  # 邮件的收件人
    msg.set_content(body)  # 使用set_content()方法设置邮件的主体内容

    # 为了防止忘记关闭连接也可以使用with语句
    with smtplib.SMTP_SSL("smtp.163.com", 465, context=context) as smtp:  # 完成加密通讯

        # 连接成功后使用login方法登录自己的邮箱
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD[:-2])

        '''
        # 方式一：
        # 使用 sendmail 方法实现发送邮件信息
        sender = EMAIL_ADDRESS      # 邮件发件人
        receive = EMAIL_ADDRESS       # 邮件收件人
        subject = "subject"
        body = "body"
        msg = f"Subject: {subject}\n\n{body}"
        print("正在发送。。。。。。。。。。。。。。。。。。")
        smtp.sendmail(sender, receive, msg)
        print("发送成功。。。。。。。。。。。。。。。。。。")
        '''

        # 方式二：使用send_message方法发送邮件信息
        smtp.send_message(msg)

if __name__ == '__main__':
    with open('./core_data.txt','r') as f:
        lines=f.readlines()
    content = ['\n'.join(lines[-8:])]
    for i in range(222,226):
        with open('./log/log9%d.txt'%i,'r') as f:
            lines=f.readlines()
            if lines[-1].strip()=='投稿失败，然后终止':
                content.append(''.join(lines))
    with open('./log/logdload.txt','r') as f:
        lines=f.readlines()
        tmp=[]
        for line in lines:
            if '视频下载成功' in line:
                tmp.append(line)
    content.append('成功下载视频数：%d\n'%len(tmp))

    send_msg('监控信息','\n'.join(content))
    print('发送成功！')