# from qibocal.calibrations.protocols import standardrb
# from qibocal.calibrations.protocols import abstract
# import plotly.graph_objects as go
# # import packages
# from plotly.subplots import make_subplots
# from qibo.noise import PauliError, NoiseModel
# from qibo import gates

# # import packages
# import numpy as np# create dummy data

# import pdb
# qubits = [0]
# depths = [1,3,5]
# runs = 10
# nshots = 100
# # Build the noise model.
# noise_params = [0.1, 0.1, 0.1]
# paulinoise = PauliError(*noise_params)
# noise = NoiseModel()
# noise.add(paulinoise, gates.Unitary)

# factory = standardrb.SingleCliffordsInvFactory(qubits, depths, runs)
# experiment = standardrb.StandardRBExperiment(factory, nshots, noisemodel=noise)
# experiment.execute()
# experiment.apply_task(standardrb.groundstate_probability)
# result = abstract.Result(experiment.dataframe)
# # xdata_scatter, ydata_scatter = result.extract('depth', 'groundstate_probabilities', list)
# xdata_scatter, ydata_scatter = result.df['depth'].to_numpy(), result.df['groundstate_probabilities'].to_numpy()
# xdata, ydata = result.extract('depth', 'groundstate_probabilities', 'mean')

# plot1 = go.Scatter(
#     x=xdata,
#     y=ydata,
#     mode='markers'
# )

# plot2 = go.Scatter(
#     x=xdata_scatter,
#     y=ydata_scatter,
#     mode='markers'
# )

# pdb.set_trace()


# # create figure
# fig = make_subplots(rows=2, cols=2)

# # plot data
# fig.add_trace(
#    plot1,
#     row=1, col=1
# )
# fig.add_trace(
#    plot1,
#     row=1, col=2
# )
# fig.add_trace(
#    plot1,
#     row=2, col=1
# )
# fig.add_trace(
#    plot2,
#     row=2, col=2
# )

# fig.update_xaxes(title_font_size=18, tickfont_size=16)
# fig.update_yaxes(title_font_size=18, tickfont_size=16)
# fig.update_layout(
#     font_family="Averta",
#     hoverlabel_font_family="Averta",
#     title_text="Subplots with Bar, Histogram, Bubble, and Box Plots",
#     xaxis1_title_text="x",
#     yaxis1_title_text="y",
#     xaxis2_title_text="Values",
#     yaxis2_title_text="Count",
#     xaxis3_title_text="x",
#     yaxis3_title_text="y",
#     xaxis4_title_text="Dataset",
#     yaxis4_title_text="Values",
#     hoverlabel_font_size=16,
#     showlegend=False,
#     height=800,
#     width=1000
# )
# fig.show()


# # result = postprocess(experiment)


# # class Myclass():
# #     def __init__(self, maxthinking: int = 3) -> None:
# #         self.maxthinking = maxthinking
# #         self.embed_function = None

# #     def __iter__(self) -> None:
# #         self.thinking = 1
# #         return self

# #     def __next__(self) -> None:
# #         if self.thinking >= self.maxthinking:
# #             raise StopIteration
# #         else:
# #             result = 'thinking of ' * self.thinking
# #             self.thinking += 1
# #             if self.embed_function is not None:
# #                 return self.embed_function(result)
# #             else:
# #                 return result

# # # def weird_function():
# # #     return 'thinking of'
# # # print(test_decorator(weird_function)())

# # Iam = Myclass(5)
# # def embed(gstring):
# #     return f'I am {gstring}cookies.'
# # # Iam.embed_function = embed
# # for word in Iam:
# #     print(word)


from qibo import gates, models

circuit = models.Circuit(2)
circuit.add(gates.H(0))
circuit.add(gates.X(1))
# print(circuit.draw())
newcircuit = models.Circuit(3)
newcircuit.add(circuit.on_qubits(*[2, 1]))


print(newcircuit.draw())
