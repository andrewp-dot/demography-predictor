import os
from model_experiments.config import LocalModelBenchmarkSettings
from abc import abstractmethod
from matplotlib.figure import Figure
from typing import List, Union

from src.local_model.model import CustomModelBase
from src.local_model.model import BaseRNN
from src.local_model.finetunable_model import FineTunableLSTM
from src.base import RNNHyperparameters

# Get settings
settings = LocalModelBenchmarkSettings()


class BaseExperiment:

    README_TEMPLATE_HEAD: str = """
# {experiment_name}

**Description:** {description}

"""

    README_PLOT: str = """
## {plot_name}
{plot_description}

![{plot_name}]({url})

"""

    def __init__(
        self,
        name: str,
        description: str,
    ):
        """
        Initializes the base attributes for the experiment.

        Args:
            name (str): Name of the experiment.
            description (str): Brief description of the experiment.
        """

        # Initiliallize base experiment settings
        self.name: str = name
        self.description: str = description

        # Save experiment path
        self.save_dir: str = settings.benchmark_results_dir
        self.experiment_dir: str = os.path.join(self.save_dir, self.name)
        self.plot_dir: str = os.path.join(self.experiment_dir, "plots")
        self.readme_path: str | None = None

        # Create directories
        self.__setup_directories()

    def __setup_directories(self) -> None:
        """
        Creates directories for the experiment. (If directories do not exist)
        """

        # Create save directory if it does not exist
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        # Create experiment directory if it does not exist
        if not os.path.isdir(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        # Create plot directory if it does not exist
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

    def create_readme(self) -> None:
        """
        Creates README.md file for the experiment.
        """

        # Create the header using template
        readme_header = self.README_TEMPLATE_HEAD.format(
            experiment_name=self.name, description=self.description
        )

        # Save readme path
        self.readme_path = os.path.join(self.experiment_dir, "README.md")

        with open(self.readme_path, "w") as readme:
            readme.write(readme_header)

    def save_plot(self, fig_name: str, figure: Figure) -> None:
        """
        Saves figure to experiment plots directory.

        Args:
            fig_name (str): File name which is used to save the figure.
            figure (Figure): The figure object that will be saved.
        """
        figure.savefig(os.path.join(self.plot_dir, fig_name))

    def readme_add_plot(
        self, plot_name: str, plot_description: str, fig_name: str
    ) -> None:
        """
        Adds the plot section to the README.md

        Args:
            plot_name (str): Name of the plot / title of the section
            plot_description (str): Description of the plot. Should describe, what is displayed on the plot.
            url (str): Url to the stored plot.

        Raises:
            ValueError: Signal that the README.md for the experiment is not created yet.
        """
        readme_plot = self.README_PLOT.format(
            plot_name=plot_name,
            plot_description=plot_description,
            url=os.path.join(".", "plots", fig_name),
        )

        # Open readme
        if self.readme_path is not None:
            with open(self.readme_path, "a") as readme:
                readme.write(readme_plot)

            return

        raise ValueError(
            "Cennot add plot! The readme does not exist! Try to use 'experiment.create_readme()' first."
        )

    def readme_add_section(self, text: str, title: str | None = None) -> None:
        """
        Add custom markdown text to the README.md

        Args:
            text (str): _description_
            title (str | None, optional): Title of the section. Please write it with '#' (markdown). Defaults to None.

        Raises:
            ValueError:  Signal that the README.md for the experiment is not created yet.
        """

        if self.readme_path is not None:

            with open(self.readme_path, "a") as readme:
                # Write title it there is some
                if title is not None:
                    readme.write(title)
                    readme.write("\n")

                # Write text
                readme.write(text)
                readme.write("\n")

            return

        raise ValueError(
            "Cannot add section! The readme does not exist! Try to use 'experiment.create_readme()' first."
        )

    def readme_add_params(
        self, custom_params: RNNHyperparameters | None = None
    ) -> None:

        write_params = self.model.hyperparameters

        if self.readme_path is not None:

            if custom_params:
                write_params = custom_params

            with open(self.readme_path, "a") as readme:
                readme.write("## Hyperparameters\n")
                readme.write(f"\n```{write_params}```\n\n")
            return

        raise ValueError(
            "Cannot add parameters! The readme does not exist! Try to use 'experiment.create_readme()' first."
        )

    def readme_add_features(self) -> None:

        if self.readme_path is not None:

            with open(self.readme_path, "a") as readme:
                readme.write("## Features\n")
                readme.write("```\n" + "\n".join(self.FEATURES) + "\n```\n\n")
            return

        raise ValueError(
            "Cannot add parameters! The readme does not exist! Try to use 'experiment.create_readme()' first."
        )

    @abstractmethod
    def run(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"Experiment '{self.name}' does not have implemented the run function!"
        )


# Hardcode the experiments
class Experiment:

    def __init__(
        self,
        name: str,
        model: BaseRNN,
        features: List[str],
        hyperparameters: RNNHyperparameters,
        experiment: BaseExperiment,
    ):
        self.name: str = name
        self.model: BaseRNN = model
        self.hyperparameters: RNNHyperparameters = hyperparameters
        self.FEATURES: List[str] = features
        self.exp: BaseExperiment = experiment

    @abstractmethod
    def run(self, state: str = "Czechia", split_rate: float = 0.8, *args, **kwargs):
        raise NotImplementedError("No run method is implemented for this experiment!")
