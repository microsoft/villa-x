import torch


class DescribeMetric:
    def __init__(
        self,
        dim: int,
        max_reservoir_size: int = 10000,
        device: str = "cpu",
    ):
        self.device = device
        self.dim = dim
        self.max_reservoir_size = max_reservoir_size
        self.max = torch.tensor(-float("inf"), device=device).float()
        self.min = torch.tensor(float("inf"), device=device).float()
        self.mean = torch.zeros(dim, device=device).float()
        self.M2 = torch.zeros(dim, device=device).float()
        self.reservoirs = [[] for _ in range(dim)]
        self.count = 0

    def update(self, val: torch.Tensor):
        val = val.to(self.device)
        self.min = torch.minimum(self.min, val.min(dim=0).values)
        self.max = torch.maximum(self.max, val.max(dim=0).values)

        n = val.shape[0]
        batch_mean = val.mean(dim=0)
        delta = batch_mean - self.mean
        new_count = self.count + n

        if n > 1:
            self.M2 += (
                val.var(dim=0, unbiased=True) * n
                + (delta**2) * self.count * n / new_count
            )
        else:
            self.M2 += (delta**2) * self.count / new_count
        self.mean += delta * (n / new_count)
        self.count = new_count

        self._update_reservoirs(val)

    def reset(self):
        self.max = torch.tensor(-float("inf"), device=self.device).float()
        self.min = torch.tensor(float("inf"), device=self.device).float()
        self.mean = torch.zeros(dim, device=self.device).float()
        self.M2 = torch.zeros(dim, device=self.device).float()
        self.reservoirs = [[] for _ in range(dim)]
        self.count = 0

    def _update_reservoirs(self, val: torch.Tensor):
        val = val.detach().cpu()
        _, D = val.shape

        for d in range(D):
            col = val[:, d]
            res = self.reservoirs[d]
            if len(res) < self.max_reservoir_size:
                res.extend(col.tolist())
            else:
                for v in col:
                    j = torch.randint(0, self.count, (1,)).item()
                    if j < self.max_reservoir_size:
                        res[j] = v.item()

    def _compute_quantiles(self, quantiles: list[float]):
        quant = []
        for d in range(self.dim):
            res = self.reservoirs[d]
            if not res:
                quant.append(torch.zeros(len(quantiles)))
            sorted_res = sorted(res)
            n = len(sorted_res)
            quant.append(
                torch.tensor([float(sorted_res[int(q * (n - 1))]) for q in quantiles])
            )
        return torch.stack(quant)

    def finalize(self, quantiles: list[float] = [0.01, 0.99]):
        std = torch.sqrt(self.M2 / max(self.count, 1))
        quant = self._compute_quantiles(quantiles)
        quant = {f"q{int(q * 100):02d}": quant[:, i] for i, q in enumerate(quantiles)}
        if self.dim > 1:
            return {
                "mean": self.mean,
                "std": std,
                "min": self.min,
                "max": self.max,
                "count": self.count,
                **quant,
            }
        else:
            return {
                "mean": self.mean.item(),
                "std": std.item(),
                "min": self.min.item(),
                "max": self.max.item(),
                "count": self.count,
                **{k: v.item() for k, v in quant.items()},
            }


if __name__ == "__main__":
    dim = 7

    metric = DescribeMetric(dim=dim, device="cpu")
    for _ in range(100):
        batch = torch.randn(32, 7) * torch.tensor(
            [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        ) + torch.tensor(
            [
                0.0,
                3.5,
                3.0,
                2.5,
                2.0,
                1.5,
                1.0,
            ]
        )
        metric.update(batch)

    print(metric.finalize())
