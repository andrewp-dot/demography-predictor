import os
from config import Config
from abc import abstractmethod
from matplotlib.figure import Figure


# Get settings
settings = Config()


class BaseExperiment:

    README_TEMPLATE_HEAD: str = """
# {experiment_name}

**Description:** {description}

"""

    README_PLOT: str = """
# {plot_name}
{plot_description}

![{plot_name}]({url})

"""

    def __init__(
        self,
        name: str,
        description: str,
    ):

        # Initiliallize base experiment settings
        self.name: str = name
        self.description: str = description

        # Save experiment path
        self.save_dir: str = settings.benchmark_results_dir
        self.experiment_dir: str = os.path.join(self.save_dir, self.name)
        self.plot_dir: str = os.path.join(self.experiment_dir, "plots")
        self.readme_path: str | None = None

        # Create directories
        self.setup_directories()

    def setup_directories(self) -> None:
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

    @abstractmethod
    def run(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"Experiment '{self.name}' does not have implemented the run function!"
        )
