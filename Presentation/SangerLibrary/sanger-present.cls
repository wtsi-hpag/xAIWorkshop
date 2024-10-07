%%
%% This is file `sanger-present.cls'
%% It sits on top of the standard beamer library 
%%
%% Licence: GPL
\ProvidesClass{sanger-present}[2023/02/07 v1.0 sanger-present class]
\NeedsTeXFormat{LaTeX2e}[1995/06/01]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{beamer}} 
\ProcessOptions\relax 
\LoadClass[aspectratio=169]{beamer} %%I fix the aspect ratio here because otherwise the images on the title page don't line up
\typeout{sanger-present: Jack Fraser-Govil, 2023}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Packages
%-------------------------------------------------------------------------------
% The only ``weird'' dependency of this package is pgf. All the rest should be
% installed on any decent system.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\typeout{Use Packages}
\RequirePackage{fontspec} %% needed to use custom fonts - this forces you to compile with XeLaTeX

%% tikz is the drawing/plotting library - these are used to construct the nice title page
\RequirePackage{tikz}
\usetikzlibrary{calc}
\usetikzlibrary{positioning}
\usepackage{lmodern}

%load the default Sanger Stuff - colours/fonts etc
\usepackage{SangerLibrary/SangerResources} 
\beamertemplatenavigationsymbolsempty

%%%%%%%%%%%%%%%%%%%
% Presentation Stuff
%Disable beamer fonts and set main font theme to Helvetica
\usefonttheme{serif}
\Helvetica{} 

%%Set Fonts for the Titlepage
\setbeamerfont{institute}{size=\footnotesize}
\setbeamerfont{date}{size=\footnotesize}
\setbeamercolor{title}{fg=SangerLight}
\setbeamercolor{subtitle}{fg=SangerLight}
\setbeamercolor{author}{fg=SangerLight}
\setbeamercolor{frametitle}{fg=SangerDark}
\setbeamertemplate{frametitle}{\usebeamercolor[fg]{frametitle}\Wellcome\insertframetitle}

%%set itemize colours to be Sanger colours
\setbeamertemplate{itemize item}{\color{SangerDeep}$\blacktriangleright$}
\setbeamertemplate{itemize subitem}{\color{SangerMedium}$\blacktriangleright$}
\setbeamertemplate{itemize subsubitem}{\color{SangerLight}$\blacktriangleright$}

%% Define the default logo
\logo{\SangerLogoColour}
\def\SangerLoc{../../Presentations/SangerResource}
%%% define the custom title page
\defbeamertemplate*{title page}{customized}[1][]
{
    {
    \setbeamertemplate{logo}{}
		\begin{tikzpicture}
      %%white squares
			\node[opacity=0, anchor=north east, at=(current page.north east),inner sep=0pt] (a) {{{\vspace{0cm}\includegraphics[height=0.8\paperwidth,angle=-90]{\SangerLoc/white_squares.pdf}}}};
      
      %%logo
      \node[below right ={-0.04\paperwidth} and {-0.07\paperwidth} of a, anchor = south east] (b) {\SangerLogoBlue};
			
      %% text in bottom left of screen
      \node[ anchor=south west, at=(current page.south west),inner sep=5pt]
      {
		    \begin{minipage}[t][0.8715\textheight]{0.6\textwidth}

	        \vspace{\fill}

          \usebeamerfont{title}\usebeamercolor*[fg]{title}{\parbox[t]{\linewidth}{\Large \raggedright \Wellcome \inserttitle}}\par
          \bigskip
          \usebeamerfont{author}{\large\insertauthor}\par
          \usebeamerfont{institute}{\insertinstitute}\par
          \usebeamerfont{date}{\insertdate}\par
		
      	\end{minipage}
			};
			\node[anchor=south east] at ({\paperwidth-1.1cm},{1.7cm}) {\usebeamercolor[fg]{titlegraphic}\inserttitlegraphic};
		\end{tikzpicture}
    }
	% }
 
}


%%%%%%%%%%%
%Custom Commands

%removes logo  - if used in braces (see maketitlepage), removes it only from that page
\newcommand{\nologo}{\setbeamertemplate{logo}{}} 

%combines \pause and \item to make sequential lists earlier
\newcommand\pitem{\pause\item}


%%print the custom title page - includes removal of logo and custom backgrounds
\newcommand\maketitlepage
{
  {
    \nologo{}
    \setbeamercolor{background canvas}{bg=SangerDark}
    \frame{\titlepage}
  }  
}

