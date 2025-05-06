import torch
import neuralprocesses.torch as nps
from attrdict import AttrDict

class CONVCNP(torch.nn.Module):
    def __init__(self, dim_x=2, dim_y=1, likelihood="het"):

        super(CONVCNP, self).__init__()
        self.likelihood = likelihood
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        self.model = nps.construct_convgnp(dim_x=dim_x, dim_y=dim_y, likelihood=self.likelihood)
        
    def forward(self, batch):

        xc, yc = batch.xc, batch.yc
        xt, yt = batch.xt, batch.yt

        # Adjusted dimensions to match nps requirements
        xc = xc.permute(0, 2, 1)
        xt = xt.permute(0, 2, 1)
        yc = yc.permute(0, 2, 1)
        yt = yt.permute(0, 2, 1)

        outs = AttrDict()
        # print(xc.shape)
        outs.tar_ll = nps.loglik(self.model, xc, yc, xt, yt, normalise=True).mean()
        outs.loss = -outs.tar_ll
        
        return outs
    # def predict(self, xc, yc, xt, num_samples=1):

    #     self.eval()
    #     with torch.no_grad():
    #         xc = xc.permute(0, 2, 1)
    #         xt = xt.permute(0, 2, 1)
    #         yc = yc.permute(0, 2, 1)

    #         mean, var, _, _ = nps.predict(self.model, xc, yc, xt)

    #         return AttrDict(
    #             loc=mean.permute(0, 2, 1),
    #             scale=var.sqrt().permute(0, 2, 1)
    #         )

    def predict(self, xc, yc, xt, num_samples=1):
        self.eval()
        with torch.no_grad():
            xc = xc.permute(0, 2, 1)
            xt = xt.permute(0, 2, 1)
            yc = yc.permute(0, 2, 1)

            # Wrap context as a list of tuples
            mean, var, _, _ = nps.predict(self.model, [(xc, yc)], xt, num_samples=num_samples)

            return AttrDict(
                mean=mean.permute(0, 2, 1),
                scale=var.sqrt().permute(0, 2, 1)
            )
