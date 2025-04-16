from pathlib import Path

import mlflow
from omegaconf import DictConfig


class MlflowExperimentManager:
    """MLflowの実験管理を行うクラス

    Attributes:
        experiment_name (str): 実験名
    """

    def __init__(self, experiment_name: str) -> None:
        """MLflowの実験管理クラスを初期化する

        Args:
            experiment_name(str): 実験名
        """
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            mlflow.create_experiment(name=experiment_name)
        mlflow.set_experiment(experiment_name)
        mlflow.enable_system_metrics_logging()  # type: ignore[no-untyped-call]

    def log_param_from_omegaconf_dict(
        self, params: DictConfig, prefix: str = ""
    ) -> None:
        """OmegaConfの設定をMLflowにログとして記録する。

        DictConfigの場合は.をつけて構造を保持して記録する。

        Args:
            params(DictConfig): OmegaConfの設定
            prefix(str): パラメータ名の接頭辞
        """
        for k, v in params.items():
            if isinstance(v, DictConfig):
                self.log_param_from_omegaconf_dict(v, prefix=f"{prefix}{k}.")  # type: ignore[str-bytes-safe]
            else:
                mlflow.log_param(f"{prefix}{k}", v)  # type: ignore[str-bytes-safe]

    def log_artifact(self, local_path: Path) -> None:
        """パスからアーティファクトをログとして記録する

        Args:
            local_path(Path): ログとして記録するパス
        """
        mlflow.log_artifact(str(local_path))

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """メトリクスをログとして記録する

        Args:
            key(str): メトリクス名
            value(float): メトリクスの値
            step(int | None): ステップ数
        """
        mlflow.log_metric(key, value, step=step)
