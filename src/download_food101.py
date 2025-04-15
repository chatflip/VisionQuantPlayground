import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path


def progress(block_count: int, block_size: int, total_size: int) -> None:
    """ダウンロードの進捗状況を表示するための関数

    Args:
        block_count (int): ダウンロードされたブロックの数
        block_size (int): 各ブロックのサイズ（バイト）
        total_size (int): ファイルの合計サイズ（バイト）
    """
    percentage = min(int(100.0 * block_count * block_size / total_size), 100)
    bar = "[{}>{}]".format("=" * (percentage // 4), " " * (25 - percentage // 4))
    sys.stdout.write("{} {:3d}%\r".format(bar, percentage))
    sys.stdout.flush()


def download_file(baseurl: str, filename: str, working_dir: Path) -> None:
    """指定されたURLからファイルをダウンロードする

    Args:
        baseurl (str): ダウンロード元のベースURL
        filename (str): ダウンロードするファイル名
        working_dir (Path): ファイルを保存するディレクトリパス

    Raises:
        urllib.error.HTTPError: HTTPリクエストに失敗した場合
        OSError: ファイルシステム関連のエラーが発生した場合
    """
    fullpath = working_dir / filename
    target_url = f"{baseurl.rstrip('/')}/{filename}"
    print(f"Downloading: {target_url}")
    try:
        urllib.request.urlretrieve(
            url=target_url, filename=str(fullpath), reporthook=progress
        )
        print("")
    except urllib.error.HTTPError as err:
        print(f"HTTPエラー: {err.code} - {err.reason}")
        raise
    except OSError as err:
        print(f"OSエラー: {err.strerror}")
        raise


def decompress_file(working_dir: Path, filename: str) -> None:
    """tar.gzファイルを解凍する

    Args:
        working_dir (Path): 解凍するファイルが存在するディレクトリパス
        filename (str): 解凍するファイル名

    Raises:
        tarfile.TarError: 解凍処理中にエラーが発生した場合
        OSError: ファイルシステム関連のエラーが発生した場合
    """
    try:
        with tarfile.open(working_dir / filename, "r:gz") as tr:
            tr.extractall(path=working_dir)
    except (tarfile.TarError, OSError) as err:
        print(f"解凍エラー: {str(err)}")
        raise


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパースする

    Returns:
        argparse.Namespace: パースされたコマンドライン引数
    """
    parser = argparse.ArgumentParser(
        description="データセットのダウンロードと解凍を行うスクリプト"
    )
    parser.add_argument(
        "--baseurl",
        type=str,
        default="http://data.vision.ee.ethz.ch/cvl",
        help="ダウンロード元のベースURL",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="food-101.tar.gz",
        help="ダウンロードするファイル名",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/datasets"),
        help="出力先ディレクトリのパス",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    download_file(args.baseurl, args.filename, args.output)
    decompress_file(args.output, args.filename)

    (args.output / args.filename).unlink()
