import requests
import json

class EverydayFunctions():
    def __init__(self):
        return
    
    # slackにメッセージを送信する
    def send_slack_message(self, message):
        # acaicalen@gmail.comアカウントのslackURL
        WEB_HOOK_URL = 'https://hooks.slack.com/services/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'
        requests.post(WEB_HOOK_URL, data=json.dumps({
            'text' : message
        }))
        return
def main():
    ef = EverydayFunctions()
    ef.send_slack_message('プログラムの実行が終了しました')
    return

if __name__ == '__main__':
    main()