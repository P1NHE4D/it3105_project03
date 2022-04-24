from abc import ABC, abstractmethod


class Domain(ABC):

    @abstractmethod
    def get_init_state(self):
        pass

    @abstractmethod
    def get_current_state(self):
        pass

    @abstractmethod
    def get_child_state(self, action):
        """
        :param action: picked action
        :return: encoded child state and corresponding reward
        """
        pass

    @abstractmethod
    def is_current_state_terminal(self):
        pass

    @abstractmethod
    def visualise(self):
        pass
