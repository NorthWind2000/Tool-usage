# 一、使用pip 和conda的区别

## 1.1 conda

首先，**conda是一个通用的包管理器**，意思是什么语言的包都可以用其进行管理，自然也就包括Python了。在安装Anaconda或者Miniconda时，会对conda进行一同安装。其中Anaconda作为Python的发行版，相当于在Python的基础上自带了常用第三方库，而Miniconda则相当于是一个conda环境的安装程序，只包含了conda及其依赖项，这样就可以减少一些不需要的第三方库的安装，所以Miniconda所占用的空间很小。

## 1.2 pip

Pip同conda一样，也是一个包管理器，并且是Python官方认可的包管理器。其中pip的含义是Pip Installs Packages。最常用于安装在Python包索引（PyPI, Python Package Index https://pypi.python.org/pypi）上发布的包。因此，在通过`conda list`命令查看当前环境下已安装的包时，通过pip的源是pypi。

#### 1.3 conda和pip安装库的区别

- 在Anaconda中，**无论在哪个环境下**，只要通过`conda install xxx`的方式安装的库都会放在Anaconda的pkgs目录下。**这样的好处就是，当在某个环境下已经下载好了某个库，再在另一个环境中还需要这个库时，就可以直接从pkgs目录下将该库复制至新环境**。
- pip 安装的库，全部都放在了**C:\Users\ZCC\Anaconda3\Lib\site-packages**目录下。同样将`Lib\site-packages`中的文件复制到当前新环境下Lib中的第三方库中，这个过程相当于通过`pip install xxx`进行了安装）而不用重复下载。

#### 1.4 conda和pip卸载库的区别

pip是在**特定的环境**中进行库的安装，所以卸载库也是一样的道理，通过`pip uninstall xxx`就可以将该环境下`Lib\site-packages`中对应的库进行卸载了。

如果通过`conda uninstall xxx`删除当前环境下某个库时，删除的只是当前环境下site-packages目录中该库的内容，它的效果和通过`pip uninstall xxx`是一样的。如果再到另一个环境中通过`conda install xxx`下载这个库，则还是通过将pkgs目录下的库复制到当前环境。若要清空这个pkgs下的已下载库，可以通过命令`conda clean -h`进行实现。

# 二、h5py的错误

把Anaconda的pkgs目录下面的带有h5py名称的文件删除，之后使用pip install h5py进行安装。

