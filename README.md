# Learning Deep Learning Deeply: LDLD

---

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Cloning this repo](#cloning-this-repo)
  - [System setup](#system-setup)
    - [About conda](#about-conda)
  - [Development setup](#development-setup)
  - [Environment _activation_](#environment-activation)
- [Prerequisites](#prerequisites)
- [Main Tutorials](#main-tutorials)
  - [0 &nbsp; Introduction to Deep Learning](#0--introduction-to-deep-learning)
  - [1 &nbsp; Multilayer Perceptrons (MLP)](#1--multilayer-perceptrons-mlp)
  - [2 &nbsp; Convolutional Neural Network (CNN)](#2--convolutional-neural-network-cnn)
  - [3 &nbsp; Recurrent Neural Networks (RNN)](#3--recurrent-neural-networks-rnn)
  - [4 &nbsp; Transformers](#4--transformers)
  - [5 &nbsp; Graph Neural Networks (GNN)](#5--graph-neural-networks-gnn)
- [Further Reading](#further-reading)
- [Comments](#comments)

---

## Introduction

Let's learn Deep Learning deeply!

## Getting Started

### Cloning this repo

Open terminal on a UNIX-like system (Linux, macOS, etc.). Then clone this
repository by:

```bash
git clone git@github.com:seoklab/ldld.git
```

Also, don't forget to change the working directory.

```bash
cd ldld
```

### System setup

**CAUTION FOR ADVANCED USERS**: The listed commands will create a new
environment named `ldld` and do nothing if it exists. Please check for existence
and remove it or check the packages in the original one against the
[environment file](environment.yml).

Two scripts are provided for:

- **Lab members**: first clone this repo on your **local machine**, and then
  simply run the following command.

  > Don't like executing suspicious scripts on your personal computer? Please
  > refer to
  > [the preface](https://github.com/seoklab/ldld-internal/blob/main/docs/before_starting.md)
  > for details.

  ```bash
  ./scripts/init_seoklab.sh
  ```

  For executing jupyter notebooks on the compute nodes, please ssh into one of
  the clusters<sup id="a1">[1](#f1)</sup>, then
  [clone this repo](#cloning-this-repo) again.

- **Individuals**: Run this command to install [`conda`](https://conda.io) and
  setup an _environment_, which will be used throughout this repo. Once the
  script does its job, you're almost done, at least the "hard" (i.e., terminal)
  part!

  ```bash
  ./scripts/bootstrap.sh
  source "$HOME/.$(basename "$SHELL")rc"
  ```

#### About conda

`conda` helps us manage python packages much easier, by creating isolated
environments<sup id="a2">[2](#f2)</sup> for each project. You'd love it as soon
as you get into the "real world" Python development. One drawback of the package
manager is that it takes **some** time to download the required packages (at
least for the first time). Both scripts will take around ~30 minutes to
complete.

### Development setup

There are two options just waiting for your choice:

1. (**Recommended**) Download [vscode](https://code.visualstudio.com) and
   install it. If you're on a Linux-based machine with sudo access, it's not a
   bad idea to give snap a chance with `$ sudo snap install code`. Once the
   editor is ready to go, setup these extensions:

   - [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
   - [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
   - [Markdown All in One](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
   - [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one)
     (for remote development<sup id="a3">[3](#f3)</sup>)

   and also take a look at these optional but useful extensions:

   - For a powerful language server:
     [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)
   - For better editor suggestions:
     [Visual Studio IntelliCode](https://marketplace.visualstudio.com/items?itemName=VisualStudioExptTeam.vscodeintellicode)
   - For beautiful (IMO<sup id="a4">[4](#f4)</sup>) color scheme:
     [Noctis](https://marketplace.visualstudio.com/items?itemName=liviuschera.noctis)

   **One extra setting left for lab members**: search vscode settings named
   `Remote.SSH: Lockfiles in Tmp`, then check it. It is required to use vscode
   on the compute nodes.

2. Start a jupyter kernel with `$ jupyter notebook` at the project root, open
   the web browser, then connect to
   [`http://localhost:8888`](http://localhost:8888)<sup id="a5">[5](#f5)</sup>.
   Local port forwarding (`ssh -L8888:localhost:8888`) might be required if the
   jupyter kernel runs on a remote machine.

### Environment _activation_

Type this command, which will be your friend while working in this repo. Type it
**every** time before starting a jupyter session. `(ldld)` will then appear in
front of the
[prompt](https://en.wikipedia.org/wiki/Command-line_interface#Command_prompt).

```bash
conda activate ldld
```

If you're using vscode (see [the previous section](#development-setup)), choose
the environment named `ldld` while starting the kernel. The editor will open a
dialog for the selection.

## Prerequisites

- Basic university math, including multivariable calculus.
- Basic shell commands: refer to the related
  [lecture](http://seoklab.org/forum/index.php?topic=5657) and a
  [blog post](https://www.44bits.io/ko/post/linux-and-mac-command-line-survival-guide-for-beginner).
- Python programming: refer to the related
  [lecture](http://seoklab.org/forum/index.php?topic=5613).

**NOTE**: Currently, the lectures are only for lab members. Only Korean versions
are available.

## Main Tutorials

**TIP**: Text too big in vscode? Get to
`Settings - Notebook - Markup: Font Size` and change it to the desired value
(`15` is a good start).

### 0 &nbsp; Introduction to Deep Learning

> what is "deep learning?"

- Handout: **TBA**
- [Notebook](notebooks/0-idl.ipynb)

### 1 &nbsp; Multilayer Perceptrons (MLP)

> a.k.a. fully connected layers, linear layers, affine layers, feed-forward
> networks, and ...

- [Handout](handouts/1-mlp.pptx)
- [Notebook](notebooks/1-mlp.ipynb)

### 2 &nbsp; Convolutional Neural Network (CNN)

> not to be confused with the prominent American news network.

- [Handout](handouts/2-cnn.pptx)
- [Notebook](notebooks/2-cnn.ipynb)

### 3 &nbsp; Recurrent Neural Networks (RNN)

> including: Long-Short Term Memory (LSTM), Gated Recurrent Unit (GRU), ...

- [Handout](handouts/3-rnn.pptx)
- [Notebook](notebooks/3-rnn.ipynb)

### 4 &nbsp; Transformers

> Attention!

- [Handout](handouts/4-trs.pptx)
- [Notebook](notebooks/4-trs.ipynb)

### 5 &nbsp; Graph Neural Networks (GNN)

> shortest Wikipedia article; longest list of proposed models: message passing
> neural networks (MPNN), graph transformers, etc., are all included in this
> category.

- [Handout](handouts/5-gnn.pptx)
- [Notebook](notebooks/5-gnn.ipynb)

## Further Reading

**TBA** <!--  -->

## Comments

<span id="f1">[1](#a1)</span> &nbsp; Contact to the system administrator if
you're not sure what you're doing.  
<span id="f2">[2](#a2)</span> &nbsp; An isolated "environment" for specific
versions of python and packages.  
<span id="f3">[3](#a3)</span> &nbsp; If not sure, consider installing it. It
wouldn't hurt!  
<span id="f4">[4](#a4)</span> &nbsp; Abbreviation of _In My Opinion_. Frequently
used in developer communities.  
<span id="f5">[5](#a5)</span> &nbsp; `8888` is the default port for jupyter
servers. It may depend on configuration and current port usage.

---

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">
<img alt="Creative Commons License" style="border-width:0"
	src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />
This work is licensed under a
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">
	Creative Commons Attribution-ShareAlike 4.0 International License</a>.

Copyright &copy; 2022- Seoul National University Lab of Computational Biology
and Biomolecular Engineering.
