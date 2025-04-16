from logging import getLogger

logger = getLogger(__name__)


class AverageMeter(object):
    """平均値と現在値を計算して保持するクラス

    Attributes:
        name (str): メーターの名前
        fmt (str): 表示フォーマット
        val (float): 現在の値
        avg (float): 平均値
        sum (float): 合計値
        count (int): カウント数
    """

    def __init__(self, name: str, fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        """メーターの値をリセットする"""
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        """メーターの値を更新する

        Args:
            val (float): 更新する値
            n (int, optional): 更新回数. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        """メーターの文字列表現を返す

        Returns:
            str: フォーマットされた文字列
        """
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """進捗状況を表示するためのメータークラス

    Attributes:
        batch_fmtstr (str): バッチ表示用のフォーマット文字列
        meters (list[AverageMeter]): 表示するメーターのリスト
        prefix (str): 表示時のプレフィックス
    """

    def __init__(
        self, num_batches: int, meters: list[AverageMeter], prefix: str = ""
    ) -> None:
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        """現在の進捗状況を表示する

        Args:
            batch (int): 現在のバッチ番号
        """
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        """バッチ表示用のフォーマット文字列を生成する

        Args:
            num_batches (int): バッチの総数

        Returns:
            str: フォーマット文字列
        """
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
