"""
DonutCode デコーダ単体テストスクリプト

【使い方】
python decoder_test.py <画像ファイルのパス> [--config コンフィグ名]

例:
python decoder_test.py sample-result/test_fresh_donut.png
python decoder_test.py my_photo.jpg --config D-27-13
"""

import sys
import os
import argparse

# パッケージ構成に合わせて import を調整してください
from donutcode import Decoder

def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="DonutCodeの画像を読み込んでデコードします。")
    parser.add_argument("image_path", help="デコード対象の画像ファイルのパス")
    parser.add_argument("--config", default="D-27-13", help="使用するコンフィグタイプ (デフォルト: D-27-13)")
    
    args = parser.parse_args()

    # ファイルの存在確認
    if not os.path.exists(args.image_path):
        print(f"❌ エラー: 画像ファイルが見つかりません -> {args.image_path}")
        sys.exit(1)

    print("===== DonutCode Decoder Test =====")
    print(f"・対象画像: {args.image_path}")
    print(f"・コンフィグ: {args.config}")
    print("-" * 34)

    try:
        # デコーダの初期化
        decoder = Decoder(config_type=args.config)
        
        # デコード処理の実行（内部でVisionProcessorによる補正も実行されます）
        result = decoder.decode_image(args.image_path)
        
        # 結果の出力
        if result:
            print(f"\n🎉 デコード成功！\n復元データ: 【 {result} 】\n")
        else:
            print("\n❌ デコード失敗: コードが検出できないか、データが破損しています。\n")
            
    except Exception as e:
        print(f"\n❌ 予期せぬエラーが発生しました: {e}\n")

if __name__ == "__main__":
    main()