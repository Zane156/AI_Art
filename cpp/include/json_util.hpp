#pragma once
#include <string>
#include <sstream>

// Escape a string for JSON (handles ", \, \n, \r, \t, \b, \f)
inline std::string JsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            default:   out += c;
        }
    }
    return out;
}

// Write a JSON string key-value pair: "key":"value"
inline void JsonPair(std::ostringstream& js, const std::string& key, const std::string& val) {
    js << "\"" << key << "\":\"" << JsonEscape(val) << "\"";
}

// Write a JSON number key-value pair: "key":number
template<typename T>
inline void JsonPairNum(std::ostringstream& js, const std::string& key, T val) {
    js << "\"" << key << "\":" << val;
}

// Simple JSON string unescape
inline std::string JsonUnescape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            switch (s[i + 1]) {
                case '"': out += '"'; i++; break;
                case '\\': out += '\\'; i++; break;
                case 'n': out += '\n'; i++; break;
                case 't': out += '\t'; i++; break;
                case 'r': out += '\r'; i++; break;
                default: out += s[i]; break;
            }
        } else {
            out += s[i];
        }
    }
    return out;
}

// Simple JSON value extractor (minimal, no dependency)
inline std::string JsonGet(const std::string& json, const std::string& key) {
    auto pos = json.find("\"" + key + "\":\"");
    if (pos != std::string::npos) {
        pos += key.size() + 4;
        auto end = json.find("\"", pos);
        while (end != std::string::npos && end > 0 && json[end-1] == '\\')
            end = json.find("\"", end + 1);
        return (end == std::string::npos) ? json.substr(pos) : json.substr(pos, end - pos);
    }
    pos = json.find("\"" + key + "\":");
    if (pos != std::string::npos) {
        pos += key.size() + 3;
        auto end = json.find_first_of(",}\n\r", pos);
        return (end == std::string::npos) ? json.substr(pos) : json.substr(pos, end - pos);
    }
    return "";
}
