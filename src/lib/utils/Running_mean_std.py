import torch
import torch.nn as nn

class RunningMeanStd(nn.Module):
    def __init__(self, epsilon: float = 1e-4, shape: tuple[int, ...]|None = (), device: str = "cuda:0"):
        """
        Calculates the running mean and std of a data stream using PyTorch
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        :param device: "cpu" or "cuda"
        """
        super().__init__()
        self.device = device
        self.epsilon = epsilon
        self.shape = shape

        if shape is not None:
            self.register_buffer("mean", torch.zeros(self.shape, dtype=torch.float32, device=self.device))
            self.register_buffer("var", torch.ones(self.shape, dtype=torch.float32, device=self.device))
            self.register_buffer("count", torch.tensor(self.epsilon, dtype=torch.float32, device=self.device))

    def standardize(self, x: torch.Tensor, update: bool = False):
        """
        Update the statistics with a new batch of data.

        :param x: Input tensor (batch_size, dims)
        :type x: torch.Tensor
        :param update: Boolean for update distribution
        :type update: bool
        """
        
        # Convert into tensor if it's not
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device)

        # Count shape if it's not declared
        if self.shape is None:
            self.shape = x.shape[1:]
            self.register_buffer("mean", torch.zeros(self.shape, dtype=torch.float32, device=self.device))
            self.register_buffer("var", torch.ones(self.shape, dtype=torch.float32, device=self.device))
            self.register_buffer("count", torch.tensor(self.epsilon, dtype=torch.float32, device=self.device))

        # if len(x.shape) > 2:
        #     # dims > 1 : batch + component-wise mean
        #     batch_mean = torch.mean(x, dim=(0, 1))
        #     batch_var = torch.var(x, dim=(0, 1), unbiased=False)
        #     batch_count = torch.tensor(x.shape[0] * x.shape[1], dtype=torch.float32, device=self.device)
        # else:
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = torch.tensor(x.shape[0], dtype=torch.float32, device=self.device)

        if update:
            self._update_distrubution(batch_mean, batch_var, batch_count)

        return (x - self.mean) / torch.sqrt(self.var + self.epsilon)
        
    def destandardize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Restore the standardized data to pure data
        
        x: Standardized input
        return: Real scale value (x * std + mean)
        """

        # Convert into tensor if it's not
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device)
            
        std = torch.sqrt(self.var + self.epsilon)
        
        return x * std + self.mean
    
    def _copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """

        new_object = RunningMeanStd(shape=self.mean.shape, epsilon=self.epsilon, device=self.device)
        new_object.mean.copy_(self.mean)
        new_object.var.copy_(self.var)
        new_object.count.copy_(self.count)
        return new_object

    def _combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.
        """

        self._update_distrubution(other.mean, other.var, other.count)

    def _update_distrubution(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: torch.Tensor) -> None:
        """
        Updates internal stats based on external batch moments.
        """
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        new_count = batch_count + self.count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(new_count)