\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{idp-model}[2025/04/28 Classe para trabalhos Tech IDP]
\LoadClass[12pt, a4paper]{article}
\RequirePackage[a4paper, left=3cm, right=2cm, top=3cm, bottom=2cm,
headheight=1.8cm]{geometry}

\RequirePackage{eso-pic}
\RequirePackage{graphicx}
\RequirePackage{wallpaper}
\RequirePackage{anyfontsize}
\RequirePackage{fontspec}
\setmainfont{Arial}
\RequirePackage{etoolbox}
\RequirePackage{titlesec}
\RequirePackage{tikz}
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{xcolor}
\RequirePackage{setspace}
\RequirePackage{titletoc}
\RequirePackage{hyperref}
\RequirePackage[most]{tcolorbox}

\RequirePackage{cite} % Pacote para formatação numérica padrão IEEE
\bibliographystyle{IEEEtran} % Estilo de bibliografia IEEE


\RequirePackage[english]{babel}

\onehalfspacing



\RequirePackage{lipsum}

\definecolor{fundo_idp}{HTML}{005DAA}
\definecolor{titulos}{HTML}{00B3FF}

\newcommand{\@nomecurso}{}
% Comando para Ciência da Computação
\newcommand{\cienciadacomputacao}{
  \renewcommand{\@nomecurso}{Ciência da Computação}
}

% Comando para Engenharia de Software
\newcommand{\engenhariadesoftware}{
  \renewcommand{\@nomecurso}{Engenharia de Software}
}

\newcommand{\titulo}[1]{\def\titulonome{#1}}
\newcommand{\autor}[1]{\def\autornome{#1}}
\newcommand{\ano}[1]{\def\anonome{#1}}

\newcommand{\@nomeorientador}{}
\newif\iforientadora
\newcommand{\orientador}[1]{
\orientadorafalse
\renewcommand{\@nomeorientador}{#1}}
\newcommand{\orientadora}[1]{
\renewcommand{\@nomeorientador}{#1}
\orientadoratrue}


\newcommand{\membrobancai}[2]{\def\@nomemembrobancai{#1}
\def\@funcaomembrobancai{#2}}
\newcommand{\membrobancaii}[2]{\def\@nomemembrobancaii{#1}
\def\@funcaomembrobancaii{#2}}

\newcommand{\dataaprovacao}[1]{\def\@dataaprovacao{#1}}

\newcommand{\palavraschave}[1]{\def\@palavraschave{#1}}
\newcommand{\keywords}[1]{\def\@keywords{#1}}

\fancyhf{} % Limpa os cabeçalhos e rodapés padrão
\renewcommand{\headrulewidth}{0pt} % Remove a linha do cabeçalho


\fancypagestyle{paginas}{
    \fancyhf{}
   \fancyhead[L]{%
    \hspace*{-3cm} % Compensa a margem esquerda do geometry
    \includegraphics[width=\paperwidth]{configs/figuras/cabecalho.png}
  }%

  
\fancyfoot[C]{%
  \begin{tikzpicture}
    \filldraw[fill=fundo_idp, draw=fundo_idp] (0,0) circle (0.4cm);
    \node[text=white] at (0,0) {\thepage};
  \end{tikzpicture}%
}
}

\fancypagestyle{pretext}{
    \fancyhf{}
   \fancyhead[L]{%
    \hspace*{-3cm} % Compensa a margem esquerda do geometry
    \includegraphics[width=\paperwidth]{configs/figuras/cabecalho.png}
  }%
}

\tcbset{
  keywordsbox/.style={
    colback=blue!5,
    colframe=blue!5, % sem borda visível
    boxrule=0pt,
    arc=2pt,
    left=4pt,
    right=4pt,
    top=2pt,
    bottom=2pt,
    boxsep=4pt,
    before skip=10pt,
    after skip=10pt,
  }
}

\pagestyle{paginas}




\newcommand{\capaprincipal}{
    \color{white}
    \begin{titlepage}
    

    \vspace*{8.5 cm}
    \begin{flushleft}
    {\Large\expandafter\MakeUppercase\expandafter{Bacharelado em}}\\
    \vspace*{0.2cm}
    {\huge\bf\expandafter\MakeUppercase\expandafter{\@nomecurso}}

    \vspace*{2.0 cm}
        \AddToShipoutPicture*{
    \put(-3,0){
    \includegraphics[width=\paperwidth,height=\paperheight]{configs/figuras/capa_tcc.pdf}
    }}
{\Large\bf\expandafter\MakeUppercase\expandafter{\titulonome}}

    
    \end{flushleft}
    \vspace*{6 cm}
    \begin{center}
    {\Large\bf\expandafter\MakeUppercase\expandafter{\autornome}}
    \vfill
    {Brasília - DF, \anonome}
    
    \end{center}
    \end{titlepage}
    \color{black}


}

\newcommand{\folharosto}{
    \clearpage
    \thispagestyle{pretext}
    \begin{flushleft}
        {\Large\bfseries\color{fundo_idp}\MakeUppercase\autornome}

        \vspace*{3cm}
        {\Large\color{fundo_idp}\MakeUppercase\titulonome}

        \vspace*{3cm}
 
        \noindent{Trabalho de Conclusão de Curso apresentado como requisito parcial para a obtenção de grau de Bacharel em \@nomecurso, pelo Instituto Brasileiro de Ensino, Desenvolvimento e Pesquisa (IDP).}


        \vspace*{3cm}

        {\bfseries\color{fundo_idp}
        \iforientadora Orientadora 
        \else Orientador
        \fi}\\
        {\@nomeorientador}

        \vspace*{3cm}
        {Brasília - DF, \anonome}
    \end{flushleft}
}

\newcommand{\folhaaprovacao}{
    \clearpage
    \thispagestyle{pretext}
   \begin{flushleft}
        {\Large\bfseries\color{fundo_idp}\MakeUppercase\autornome}

        \vspace*{3cm}
        {\Large\color{fundo_idp}\MakeUppercase\titulonome}

    \end{flushleft}
        \vspace*{2cm}

       \noindent{Trabalho de Conclusão de Curso apresentado como requisito parcial para a obtenção de grau de Bacharel em \@nomecurso, pelo Instituto Brasileiro de Ensino, Desenvolvimento e Pesquisa (IDP).}


    \begin{flushleft}

        \vspace*{2cm}
        {Aprovado em \@dataaprovacao}

        \vspace*{1cm}
        {\bfseries\color{fundo_idp}Banca Examinadora}
        
        
        {
        \vspace*{2cm}
        \rule{\linewidth}{0.4pt}
        \vspace{1em}
        {\@nomeorientador -  \iforientadora Orientadora 
        \else Orientador
        \fi}

        \vspace*{2cm}
        \rule{\linewidth}{0.4pt}
        \vspace{1em}
        {\@nomemembrobancai - \@funcaomembrobancai}
        
        \vspace*{2cm}
        \rule{\linewidth}{0.4pt}
        \vspace{1em}
        {\@nomemembrobancaii - \@funcaomembrobancaii}
        }
    \end{flushleft}
}
\newcommand{\agradecimentos}{
    \clearpage
     \begin{flushleft}
    {\Large\bfseries\color{fundo_idp}\MakeUppercase{Agradecimentos}}
    \end{flushleft}
    \vspace{1em}
    \input{partes/agradecimentos}
}

\newcommand{\dedicatoria}{
    \clearpage
     \begin{flushleft}
    {\Large\bfseries\color{fundo_idp}\MakeUppercase{Dedicatória}}
    \end{flushleft}
    \vspace{1em}
    \input{partes/dedicatoria}
}

\newcommand{\resumo}{
    \clearpage
     \begin{flushleft}
    {\Large\bfseries\color{fundo_idp}\MakeUppercase{Resumo}}
    \end{flushleft}
    \vspace{1em}
    \input{partes/resumo}
    \vspace{1em}
     \begin{tcolorbox}[keywordsbox]
     \begin{center}
        \color{fundo_idp}\textbf{Palavras-chave:} \color{fundo_idp}{\@palavraschave}.
    \end{center}
    \end{tcolorbox}
}

\renewcommand{\abstract}{
    \clearpage
     \begin{flushleft}
    {\Large\bfseries\color{fundo_idp}\MakeUppercase{Abstract}}
    \end{flushleft}
    \vspace{1em}
    \input{partes/abstract}
    \vspace{1em}
     \begin{tcolorbox}[keywordsbox]
        \begin{center}
        \color{fundo_idp}\textbf{Keywords:} \color{fundo_idp}{\@keywords}.
        \end{center}
    \end{tcolorbox}
    \clearpage
}


\newcommand{\capafinal}{
\clearpage
\thispagestyle{empty}
 \vspace*{2.0 cm}
        \AddToShipoutPicture*{
    \put(-3,0){
    \includegraphics[width=\paperwidth,height=\paperheight]{configs/figuras/capa_final.pdf}
    }}

\clearpage
}


% Comando para a "página de abertura de seção" CORRIGIDO
\newcommand{\chapterbreak}[2]{%
    \clearpage
    \thispagestyle{empty}
    \AddToShipoutPicture*{
        \put(-3,0){
        \includegraphics[width=\paperwidth,height=\paperheight]{configs/figuras/capa_secoes.pdf}
    }}
    \color{white}
    \vspace*{\fill}
    \begin{center}
        {\fontsize{400}{480}\bfseries #1}
    \end{center}
    \vspace*{\fill}
    \clearpage
    \phantomsection
    \addcontentsline{toc}{section}{\protect\numberline{#1}#2} % CORREÇÃO PRINCIPAL
    \color{black}
}

\newcounter{secnum}

\newcommand{\secao}[2]{%
  \refstepcounter{secnum}%
  \chapterbreak{\arabic{secnum}}{#1}%
  
  \noindent
  \hspace*{-3.5cm}
  \begin{tikzpicture}
    \fill[cyan!70!blue!60] (0,0) rectangle (4,4);
    \node[white, font=\bfseries\fontsize{80}{80}\selectfont] at (2,2) {\arabic{secnum}};
  \end{tikzpicture}%
  \hspace{1em}%
  \begin{minipage}[t]{0.75\textwidth}
    \vspace{-0.8cm}
    {\color{titulos}\huge\bfseries\MakeUppercase{#1}}\par\vspace{1ex} % Adicionei {} aqui
  \end{minipage}
  \vspace{2em}


  \input{#2}
}

%%-------SUBSECOES--------------%%
\newcounter{subsecnum}[secnum]
\renewcommand{\thesubsecnum}{\arabic{secnum}.\arabic{subsecnum}}

\newcommand{\subsecao}[1]{%
  \refstepcounter{subsecnum}
  \par\vspace{1.5ex}

      \phantomsection
      \addcontentsline{toc}{subsection}{\arabic{secnum}.\arabic{subsecnum} #1} % <-- Adiciona ao sumário

  {\large\noindent\color{fundo_idp}\bfseries\MakeUppercase{\thesubsecnum\quad #1}}\par
  \vspace{1ex}
}


% Define o contador de subsubseções
\newcounter{subsubsecnum}[subsecnum]
\renewcommand{\thesubsubsecnum}{\arabic{secnum}.\arabic{subsecnum}.\arabic{subsubsecnum}}

% Define o comando para subsubseções
\newcommand{\subsubsecao}[1]{%
  \refstepcounter{subsubsecnum}
  \par\vspace{1.2ex}
  \phantomsection
  \addcontentsline{toc}{subsubsection}{\thesubsubsecnum #1} % <-- Adiciona ao sumário

  {\normalsize\noindent\color{fundo_idp}\bfseries\thesubsubsecnum\quad #1}\par
  \vspace{0.5ex}
}


% REFERENCIAS
\newcommand{\capareferencias}{
    \clearpage
    \thispagestyle{empty}
    \AddToShipoutPicture*{
        \put(-3,0){
            \includegraphics[width=\paperwidth,height=\paperheight]{configs/figuras/capa_secoes.pdf}
        }
    }
     \color{white}

    \vspace*{\fill}
    \begin{center}

    {\fontsize{60}{0}\bfseries\MakeUppercase{References}}
    \end{center}
    \vspace*{\fill}

    \refstepcounter{secnum}
    \phantomsection
    \addcontentsline{toc}{section}{References}

    \color{black}
    
    \null
    \clearpage
}

\newcommand{\referencias}{
    \capareferencias
    \bibliography{referencias} % Nome do arquivo .bib
}

\addto\captionsportuguese{
  \renewcommand{\refname}{%
    \color{titulos}\bfseries\MakeUppercase{References}
  }
}

% SUMÁRIO
% Altera o título do sumário
\addto\captionsportuguese{%
  \renewcommand{\contentsname}{\color{white}\fontsize{40}{0}\bfseries\MakeUppercase{Sumário}}
}
\addto\captionsenglish{%
  \renewcommand{\contentsname}{\color{white}\fontsize{40}{0}\bfseries\MakeUppercase{Contents}}
}


\addto\captionsportuguese{%
  \renewcommand{\listfigurename}{\color{white}\fontsize{40}{0}\bfseries\MakeUppercase{Lista de Figuras}}
  \renewcommand{\listtablename}{\color{white}\fontsize{40}{0}\bfseries\MakeUppercase{Lista de Tabelas}}
}

\addto\captionsenglish{%
  \renewcommand{\listfigurename}{\color{white}\fontsize{40}{0}\bfseries\MakeUppercase{List of Figures}}
  \renewcommand{\listtablename}{\color{white}\fontsize{40}{0}\bfseries\MakeUppercase{List of Tables}}
}




% Estilo dos itens da lista de figuras
\titlecontents{figure}[0pt]
{\vspace{0.5ex}\bfseries\large\color{white}}
{\contentslabel{2em}}
{}
{\titlerule*[0.5pc]{.}\contentspage}

% Estilo dos itens da lista de tabelas
\titlecontents{table}[0pt]
{\vspace{0.5ex}\bfseries\large\color{white}}
{\contentslabel{2em}}
{}
{\titlerule*[0.5pc]{.}\contentspage}

% Estilo do sumário
\titlecontents{section}[0pt]
{\vspace{0.5ex}\bfseries\large\color{white}}
{\contentslabel{2em}}
{}
{\titlerule*[0.5pc]{.}\contentspage}



\titlecontents{subsection}[2em]
{\vspace{0.3ex}\normalsize\color{white}}
{\contentslabel{2.3em}}
{}
{\titlerule*[0.5pc]{.}\contentspage}

\titlecontents{subsubsection}
  [4.5em] % Indentação ainda maior
  {\color{white}\addvspace{2pt}\small\itshape} % Formatação diferenciada
  {\contentslabel{3em}} % Número
  {} % Sem prefixo
  {\titlerule*[.5pc]{.}\contentspage} % Página



\newcommand{\backgroundsumario}{%
    \AddToShipoutPictureBG{%
        \put(0,0){%
            \includegraphics[width=\paperwidth,height=\paperheight]{configs/figuras/capa_secoes.pdf}%
        }%
    }%
}

\newenvironment{sumariolistas}{%
    \backgroundsumario
    \pagestyle{empty}%
}{%
    \ClearShipoutPictureBG
}

\makeatletter
% \renewcommand\tableofcontents{%
%     \cleardoublepage
%     \thispagestyle{empty} % Remove cabeçalhos e rodapés
%     \backgroundsumario % Ativa o fundo apenas nesta página
%     \section*{\contentsname} % Título do sumário
%     \@starttoc{toc} % Conteúdo do sumário
   
%     \ClearShipoutPictureBG % Remove o fundo das páginas seguintes
% }

\renewcommand\tableofcontents{%
    \cleardoublepage
   \begingroup
   \backgroundsumario
   \pagestyle{empty}
    \section*{\contentsname}
    \@starttoc{toc}
    \clearpage
    \endgroup
    \ClearShipoutPictureBG
}
\makeatother

\makeatletter
\renewcommand\listoffigures{%
    \cleardoublepage
   \begingroup
   \backgroundsumario
   \pagestyle{empty}
    \section*{\listfigurename}
    \@starttoc{lof}
     \clearpage
    \endgroup
    \ClearShipoutPictureBG
}

\renewcommand\listoftables{%
    \cleardoublepage
   \begingroup
   \backgroundsumario
   \pagestyle{empty}
    \section*{\listtablename}
    \@starttoc{lot}
     \clearpage
    \endgroup
    \ClearShipoutPictureBG
}
\makeatother

% APENDICE
\newcommand{\capaapendice}[1]{
    \clearpage
    \thispagestyle{empty}
    \AddToShipoutPicture*{
        \put(-3,0){
            \includegraphics[width=\paperwidth,height=\paperheight]{configs/figuras/capa_secoes.pdf}
        }
    }
     \color{white}

    \vspace*{\fill}
    \begin{center}

    {\fontsize{60}{0}\bfseries\MakeUppercase{#1}}
    \end{center}
    \vspace*{\fill}

    \refstepcounter{secnum}
    \phantomsection
    \addcontentsline{toc}{section}{#1}

    \color{black}
    
    \null
    \clearpage
}


\newcounter{apendicecount}
\setcounter{apendicecount}{0}%

\newcommand{\apendice}[2]{
        \clearpage
        \stepcounter{apendicecount}%
        \ifnum\value{apendicecount}=1
        \capaapendice{Appendices}
        \fi
        \phantomsection
        \addcontentsline{toc}{subsection}{\MakeUppercase{\alph{apendicecount}}\hspace{1em} #1}
        \begin{center}
        {\color{fundo_idp}\Large{Appendix \MakeUppercase{\alph{apendicecount}} - {#1}}} 
     
        \end{center}
        
        \input{#2}
}

%% Anexos


\newcounter{anexocount}
\setcounter{anexocount}{0}%

\newcommand{\anexo}[2]{
        \clearpage
        \stepcounter{anexocount}%
        \ifnum\value{anexocount}=1
        \capaapendice{Annexes}
        \fi
        \phantomsection
        \addcontentsline{toc}{subsection}{\MakeUppercase{\alph{anexocount}}\hspace{1em} #1}
        \begin{center}
        {\color{fundo_idp}\Large{Annex \MakeUppercase{\alph{anexocount}}} - {#1}}
        \end{center}
        
        \input{#2}
}


