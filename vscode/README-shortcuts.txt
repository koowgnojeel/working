https://marketplace.visualstudio.com/items?itemName=ms-python.python


Profiles

    File > Preferences > Profiles

    %APPDATA%\Code\User\profiles
    $HOME/.config/Code/User/profiles

    Profiles in Visual Studio Code
    https://code.visualstudio.com/docs/editor/profiles


panels

    top 

        Command Palette
            CTRL_SHIFT_p

    left(side bar)

        toggle side-bar
            CTRL_b

        file explorer panel
            CTRL_SHIFT_e

        extensions view (marketplace)

            CTRL_SHIFT_x

            Managing Extensions in Visual Studio Code
            https://code.visualstudio.com/docs/editor/extension-marketplace

        source control
            CTRL_SHIFT_g

        run and debug panel
            CTRL_SHIFT_d

    bottom

        toggle panel
            CTRL_j 

        output
            CTRL_K CTRL_h

        integrated terminal 
            CTRL_`

            New Terminal 
                CTRL_SHIFT_`

        toggle debug console
            CTRL_SHIFT_Y

basic

    full screen           F11
    full screen(Zen mode) CTRL_k z

    close window
        CTRL_w

    switch tab
        CTRL_PageDown/PageUp

    open file
        CTRL_o

    new tab
        CTRL_n

    zoom in/out
        CTRL_+, CTRL_-

    goto
        CTRL_g


search/replace

    Replace in files
        CTRL_SHIFT_h 

    Show Search

        CTRL_SHIFT_f

        Toggle Search details
            CTRL_SHIFT_j 


editing(coding)

    focus

        editor window
            CTRL_1

        nth tab
            ALT_1

    code completion
        CTRL_SPACE

    go to definition
        F12

    comment

        (line)  CTRL_/
        (block) CTRL_SHIFT_a

    create a virtual environment

        CTRL_SHIFT_P > Python: Create Environment

    run
        CTRL_F5

    debug

        Java/Python
            F5

        breakpoint
            F9

        step
            over
                F10

            into
                F11

            out
                SHIFT_F11

        stop
            SHIFT_F5

        resumes
            F5

    markdown
    https://code.visualstudio.com/docs/languages/markdown

        preview
            CTRL_SHIFT_v

        preview in split
            CTRL_k, v

        switch split
            CTRL_1, CTRL_2


Etc.

    Theme

        Visual Studio Code Themes
        https://code.visualstudio.com/docs/getstarted/themes

        * File > Preferences > Theme > Color Theme menu item 
        * CTRL_K CTRL_T to display the Color Theme picker.

    visual studio code - How to toggle between vim-emulation and no-vim-emulation when the vscodevim extension is installed? - Stack Overflow
    https://stackoverflow.com/questions/47502318/how-to-toggle-between-vim-emulation-and-no-vim-emulation-when-the-vscodevim-exte

        Set CTRL_SHIFT_v(Markdown preview) to toggle Vim mode on/off - works w/ vscodevim.vim

            From command palette 
                CTRL_SHIFT_p

            Type
                Preferences: Open Keyboard Shortcuts (JSON) (== keybindings.json) 

            Type
                [
                    {
                        "key": "ctrl+shift+v",
                        "command": "toggleVim"
                    }
                ]

    Copilot(NOT FREE)

        CTRL_ALT_i

        chat

            CTRL_i

