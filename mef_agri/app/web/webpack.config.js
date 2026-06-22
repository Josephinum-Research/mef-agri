const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');

// TODO have a look at the following link
// https://stackoverflow.com/questions/39798095/multiple-html-files-using-webpack

module.exports = {
    entry: './src/tab_prj.js',

    output: {
        path: path.resolve(__dirname, 'dist'),
        filename: 'bundle.js',
    },
    module: {
        rules: [
            {
                test: /\.css$/,
                use: ['style-loader', 'css-loader'],
            },
        ],
    },
    plugins: [
        new HtmlWebpackPlugin({
            template: './src/index.html',
        }),
    ],
    mode: 'development',
};