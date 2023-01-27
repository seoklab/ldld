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
  - [0   Introduction to Deep Learning](#0--introduction-to-deep-learning)
  - [1   Multilayer Perceptrons (MLP)](#1--multilayer-perceptrons-mlp)
  - [2   Convolutional Neural Networks (CNN)](#2--convolutional-neural-networks-cnn)
  - [3   Recurrent Neural Networks (RNN)](#3--recurrent-neural-networks-rnn)
  - [4   Transformers](#4--transformers)
  - [5   Graph Neural Networks (GNN)](#5--graph-neural-networks-gnn)
- [Example project layout](#example-project-layout)
- [Further Reading](#further-reading)
- [License and disclaimer](#license-and-disclaimer)

---

## Introduction

Let's learn Deep Learning deeply!

## Getting Started

### Cloning this repo

Open terminal on a UNIX-like system (Linux, macOS, etc.). Then clone this
repository by:

```bash
git clone --recurse-submodules git@github.com:seoklab/ldld.git
```

You could safely ignore the submodule related errors if you're not a lab member.

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

- **Everyone**: Run this command to install [`conda`](https://conda.io) and
  setup an _environment_, which will be used throughout this repo. Also don't
  forget to re-login after running the command. Once the script does its job,
  you're almost done, at least the "hard" (i.e., terminal) part!

  ```bash
  ./scripts/bootstrap.sh
  ```

  **NOTE**: If you're a lab member, then you must run this command on the
  cluster, not on your local machine.

- **Lab members only** (optional): first clone this repo on your **local
  machine**, and then simply run the following command. This will update ssh
  settings for convenience.

  > Don't like executing suspicious scripts on your personal computer? Please
  > refer to
  > [the preface](https://github.com/seoklab/ldld-internal/blob/main/docs/before_starting.md)
  > for details.

  ```bash
  ./scripts/init_seoklab.sh
  ```

  For executing jupyter notebooks on the compute nodes, please ssh into the
  cluster[^1], then [clone this repo](#cloning-this-repo)
  again.

[^1]: Contact to the system administrator if you're not sure what you're doing.

#### About conda

`conda` helps us manage python packages much easier, by creating isolated
environments[^2] for each project. You'd love it as soon
as you get into the "real world" Python development. One drawback of the package
manager is that it takes **some** time to download the required packages (at
least for the first time). Both scripts will take around ~30 minutes to
complete.

[^2]: An isolated "environment" for specific versions of python and packages.

### Development setup

There are two options just waiting for your choice:

1. (**Recommended**) Download [vscode](https://code.visualstudio.com) and
   install it. If you're on a Linux-based machine with sudo access, it's not a
   bad idea to give snap a try with `$ sudo snap install code --classic`. Once
   the editor is ready to go, setup these extensions:

   - [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
   - [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
   - [Markdown All in One](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
   - [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one)
     (for remote development[^3])

   and also take a look at these optional but useful extensions:

   - For a powerful language server:
     [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)
   - For better editor suggestions:
     [Visual Studio IntelliCode](https://marketplace.visualstudio.com/items?itemName=VisualStudioExptTeam.vscodeintellicode)
   - For beautiful (IMO[^4]) color scheme:
     [Noctis](https://marketplace.visualstudio.com/items?itemName=liviuschera.noctis)

2. Start a jupyter kernel with `$ jupyter notebook` at the project root, open
   the web browser, then connect to
   [`http://localhost:8888`](http://localhost:8888)[^5].
   Local port forwarding (`ssh -L8888:localhost:8888`) might be required if the
   jupyter kernel runs on a remote machine.

[^3]: If not sure, consider installing it. It wouldn't hurt!
[^4]: Abbreviation of _In My Opinion_. Frequently used in developer communities.
[^5]: `8888` is the default port for jupyter servers. It may depend on
configuration and current port usage.

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
  [lecture](https://drive.google.com/drive/folders/1rjDuxl17BgFFWa1MrTUBrElYvXTPJlZH) and a
  [blog post](https://www.44bits.io/ko/post/linux-and-mac-command-line-survival-guide-for-beginner).
- Python programming: refer to the related
  [lecture](https://drive.google.com/drive/folders/1rjDuxl17BgFFWa1MrTUBrElYvXTPJlZH).

**NOTE**: Only Korean versions are available.

## Main Tutorials

**TIP**: Text too big in vscode? Get to
`Settings - Notebook - Markup: Font Size` and change it to the desired value
(`15` is a good start).

### 0 &nbsp; Introduction to Deep Learning

> what is "deep learning?"

- [Handout](handouts/README.md)
- [Notebook](notebooks/0-idl.ipynb)

### 1 &nbsp; Multilayer Perceptrons (MLP)

> a.k.a. fully connected layers, linear layers, affine layers, feed-forward
> networks, and ...

- [Handout](handouts/1-mlp.pptx)
- [Notebook](notebooks/1-mlp.ipynb)

### 2 &nbsp; Convolutional Neural Networks (CNN)

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

## Example project layout

This repository also serves as an example project layout. The following
directories are included:

- `src/`: example source code layout for model, dataset, training, and utilities.
- `pyproject.toml` and `setup.cfg`: example project configuration, required for
  *installing* (`pip install [-e] .`) the project.

## Further Reading

**TBA** <!--  -->

## License and disclaimer

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">
<img alt="Creative Commons License" style="border-width:0;"
	src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a>

The source code examples used in this project are licensed under the
[MIT License](LICENSE.txt). Other contents of this repository are licensed under
the
[Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

Copyright &copy; 2022- Seoul National University Lab of Computational Biology
and Biomolecular Engineering.

> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.
