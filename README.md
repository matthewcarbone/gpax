# GPax: Gaussian Processes for Experimental Science

<!--main-readme-start-->

‼️  **This is an experimental fork and is not stable! Please do not use this for production science! Expect breaking changes!**


GPax is a small Python package for physics-based Gaussian processes (GPs) built on top of NumPyro and JAX. Its purpose is to take advantage of prior physical knowledge and different data modalities when using GPs for data reconstruction and active learning. It is a work in progress, and more models will be added in the near future.


![GPax_logo](https://github.com/ziatdinovmax/gpax/assets/34245227/f2117b9d-d64b-4e48-9b91-e5c7f220b866)

<!--main-readme-end-->


# Installation

link TK (docs)


# Development

You can use sphinx-autobuild to continuously rebuild the documentation:
```bash
sphinx-autobuild -b html docs/source docs/build --open-browser --watch gpax
```

# Cite us

If you use GPax in your work, please consider citing our papers:

```
@article{ziatdinov2021physics,
  title={Physics makes the difference: Bayesian optimization and active learning via augmented Gaussian process},
  author={Ziatdinov, Maxim and Ghosh, Ayana and Kalinin, Sergei V},
  journal={arXiv preprint arXiv:2108.10280},
  year={2021}
}

@article{ziatdinov2021hypothesis,
  title={Hypothesis learning in an automated experiment: application to combinatorial materials libraries},
  author={Ziatdinov, Maxim and Liu, Yongtao and Morozovska, Anna N and Eliseev, Eugene A and Zhang, Xiaohang and Takeuchi, Ichiro and Kalinin, Sergei V},
  journal={arXiv preprint arXiv:2112.06649},
  year={2021}
}
```

## Funding acknowledgment
This work was supported by the U.S. Department of Energy, Office of Science, Basic Energy Sciences Program.
