#ifndef __ARG_PARSER_HPP__
#define __ARG_PARSER_HPP__

#include <CLI/CLI.hpp>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace common {

/**
 * @brief The ArgParser class encapsulates the CLI11 library for parsing command-line parameters.
 */
class ArgParser{
public:
    /**
     * @brief constructor.
     * @param description: A description of the application.
    */
    ArgParser(const std::string& description = ""): m_parser{std::make_unique<CLI::App>(description)} {}

    virtual ~ArgParser() = default;
    ArgParser(const ArgParser&) = delete;
    ArgParser& operator=(const ArgParser&) = delete;
    ArgParser(ArgParser&&) = default;
    ArgParser& operator=(ArgParser&&) = default;


    /**
     * @brief Set the INI configuration file options.
     * @param config_name The name of the configuration option.
     * @param default_filename The default configuration file path.
     * @param help_message Help message.
     * @param config_required Whether the configuration file is required.
     */
    void setConfig(const std::string& config_name, const std::string& default_filename = "", const std::string& help_message = "Read an ini file", const bool config_required = false){
        m_parser->set_config(config_name, default_filename, help_message, config_required);
    }

    /**
     * @brief Add a command-line option.
     * @param name The name of the option (e.g., "--input").
     * @param defaultVal The default value of the option.
     * @param description The description of the option.
     */
    template <typename T>
    void addOption(const std::string& name, const T defaultVal, const std::string& description = ""){
        CLI::Option *opt = m_parser->add_option(name, description);
        if (opt){
            opt->default_val(defaultVal);
        }
    }

    /**
     * @brief Get the value of a specified option.
     * @param option_name The name of the option.
     * @param[out] ret The variable to store the option value.
     */
    template <typename T>
    void getOptionVal(const std::string& option_name, T& ret){
        CLI::Option *opt = m_parser->get_option(option_name);
        if (opt){
            ret = opt->as<T>();
        }
    }

    /**
     * @brief Add a flag (flag is an option without a value, only two states: true/false).
     * @param flag_name The name of the flag (e.g., "--verbose").
     * @param defaultVal The default value of the flag (true or false).
     * @param description The description of the flag.
     */
    void addFlag(const std::string& flag_name, const bool defaultVal, const std::string& description = ""){
        CLI::Option *flag = m_parser->add_flag(flag_name, description);
        if (flag){
            flag->default_val(defaultVal);
        }
    }

    /**
     * @brief Get the value of a specified flag.
     * @param flag_name The name of the flag.
     * @return The value of the flag (true or false).
     */
    bool getFlagVal(const std::string& flag_name){
        return m_parser->get_option(flag_name)->as<bool>();
    }

    /**
     * @brief Parse command-line arguments.
     * @param argc The number of command-line arguments.
     * @param argv The array of command-line arguments.
     * @return 0 on success, non-zero on failure.
     */
    auto parseArgs(int argc, char* argv[]) noexcept{
        try{
            m_parser->parse(argc, argv);
            return 0;
        }
        catch (const CLI::ParseError &e){
            auto exit_code = e.get_exit_code();
            if(exit_code == CLI::CallForAllHelp().get_exit_code()){
                std::cout<< m_parser->help() << std::endl;
                std::exit(exit_code);
            } else {
                return exit_code;
            }
        }
    }

    /**
     * @brief Add a subcommand.
     * @param name The name of the subcommand.
     * @param description The description of the subcommand.
     */
    void addSubCmd(const std::string& name, const std::string& description = ""){
        m_subcmdMap[name] = m_parser->add_subcommand(name, description);
    }

    /**
     * @brief Add an option to a specified subcommand.
     * @param subcmd The name of the subcommand.
     * @param name The name of the option.
     * @param defaultVal The default value of the option.
     * @param description The description of the option.
     */
    template <typename T>
    void addSubOption(const std::string& subcmd, const std::string& name, const T defaultVal, const std::string& description = ""){

        auto it = m_subcmdMap.find(subcmd);
        if(it == m_subcmdMap.end()){
            addSubCmd(subcmd);
        }
        auto subcmdParser_ptr = m_subcmdMap[subcmd];
        CLI::Option *opt = subcmdParser_ptr->add_option(name, description);
        if (opt){
            opt->default_val(defaultVal);
        }
    }

    /**
     * @brief Get the value of a specified option for a subcommand.
     * @tparam T The type of the option value.
     * @param subcmd The name of the subcommand.
     * @param option_name The name of the option.
     * @param[out] ret The variable to store the option value.
     */
    template <typename T>
    void getSubOptionVal(const std::string& subcmd, const std::string& option_name, T& ret){
        auto it = m_subcmdMap.find(subcmd);
        if(it == m_subcmdMap.end()){
            return;
        }

        auto subcmdParser_ptr = m_subcmdMap[subcmd];
        CLI::Option *opt = subcmdParser_ptr->get_option(option_name);
        if (opt){
            ret = opt->as<T>();
        }
    }

     /**
     * @brief Get the list of string options for a subcommand, using semicolon as the delimiter.
     * @param subcmd The subcommand name.
     * @param option_name The option name.
     * @param[out] ret The list of strings after splitting.
     */
    void getOptionSplitStrList(const std::string& subcmd, const std::string& option_name, std::vector<std::string>& ret){
        std::string ids_input;
        getSubOptionVal(subcmd, option_name, ids_input);
        std::stringstream ss(ids_input);
        std::string item;
        ret.clear();

        auto trim = [](std::string_view str) {
            while (!str.empty() && std::isspace(str.front())) str.remove_prefix(1);
            while (!str.empty() && std::isspace(str.back())) str.remove_suffix(1);
            return str;
        };

        while (std::getline(ss, item, ';')){
            ret.push_back(std::string(trim(item)));
        }
    }

    /**
     * @brief Get the list of string options, using semicolon as the delimiter.
     * @param option_name The option name.
     * @param[out] ret The list of strings after splitting.
     */
    void getOptionSplitStrList(const std::string& option_name, std::vector<std::string>& ret){
        std::string ids_input;
        getOptionVal(option_name, ids_input);
        std::stringstream ss(ids_input);
        std::string item;
        ret.clear();

        auto trim = [](std::string_view str) {
            while (!str.empty() && std::isspace(str.front())) str.remove_prefix(1);
            while (!str.empty() && std::isspace(str.back())) str.remove_suffix(1);
            return str;
        };

        while (std::getline(ss, item, ';')){
            ret.push_back(std::string(trim(item)));
        }
    }

    /**
     * @brief Add a flag to a specified subcommand.
     * @param subcmd The name of the subcommand.
     * @param flag_name The name of the flag.
     * @param defaultVal The default value of the flag.
     * @param description The description of the flag.
     */
    void addSubFlag(const std::string& subcmd, const std::string& flag_name, const bool defaultVal, const std::string& description = ""){
        auto it = m_subcmdMap.find(subcmd);
        if(it == m_subcmdMap.end()){
            addSubCmd(subcmd);
        }

        auto subcmdParser_ptr = m_subcmdMap[subcmd];
        CLI::Option *flag = subcmdParser_ptr->add_flag(flag_name, description);
        if (flag){
            flag->default_val(defaultVal);
        }
    }

    /**
     * @brief Get the value of a specified flag for a subcommand.
     * @param subcmd The name of the subcommand.
     * @param flag_name The name of the flag.
     * @return The value of the flag (true or false). If the subcommand doesn't exist, returns false.
     */
    bool getSubFlagVal(const std::string& subcmd, const std::string& flag_name)
    {
        auto it = m_subcmdMap.find(subcmd);
        if(it == m_subcmdMap.end()){
            return false;
        }

        auto subcmdParser_ptr = m_subcmdMap[subcmd];
        return subcmdParser_ptr->get_option(flag_name)->as<bool>();
    }


private:
    /**
     * @brief A smart pointer pointing to the CLI::App object from the CLI11 library. This is the main parser.
     */
    std::unique_ptr<CLI::App> m_parser; // Change unique_ptr to shared_ptr for CLI::App
    /**
     * @brief A hash table to store the names of subcommands and pointers to CLI::App objects (subcommands are also CLI::App objects).
     */
    std::unordered_map<std::string, CLI::App*> m_subcmdMap;
};

} // namespace common

#endif // __ARG_PARSER_HPP__
