from abc import abstractmethod


class BaseExperiment:

    @abstractmethod
    def run(self, *args, **kwargs) -> None:
        pass
